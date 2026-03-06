"""
app.py – DronRForensic
======================
Streamlit web application for drone RF-fingerprint forensics.

Workflow:
    1. Upload RF signal files (CSV / NPY / BIN / WAV) or generate synthetic ones.
    2. Compute STFT spectrograms and visualise them.
    3. Label each spectrogram (brand + model) and build a dataset.
    4. Export the dataset (HDF5 or NPZ) for re-use.
    5. Run stratified K-fold cross-validation and train a CNN.
    6. Download the trained model and use it to identify unknown drones.
"""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.dataset_handler import DatasetHandler
from src.model import build_cnn, load_model_from_bytes, save_model_to_bytes
from src.signal_processor import SignalProcessor
from src.trainer import ModelTrainer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DronRForensic",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-container { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛸 DronRForensic")
st.caption("Identificación Forense de Drones por Huella de Radiofrecuencia (RF) mediante CNN")
st.markdown("---")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "processor": SignalProcessor(),
    "dataset_handler": DatasetHandler(),
    "trained_model": None,
    "label_map": None,
    "cv_results": None,
    "current_spec": None,  # dict with processed spectrogram info
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _plot_spectrograms(f, t, spec_db, spec_norm, title_prefix: str = "") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].pcolormesh(t, f / 1e6, spec_db, shading="auto", cmap="inferno")
    axes[0].set_ylabel("Frecuencia (MHz)")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_title(f"{title_prefix}Espectrograma STFT (dB)")
    im = axes[1].imshow(spec_norm, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xlabel("Tiempo (frames)")
    axes[1].set_ylabel("Frecuencia (bins)")
    axes[1].set_title(f"{title_prefix}Espectrograma Normalizado ({spec_norm.shape[0]}×{spec_norm.shape[1]})")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def _generate_synthetic_dataset(processor: SignalProcessor, handler: DatasetHandler) -> None:
    """Populate handler with a balanced synthetic dataset (6 classes × 5 samples)."""
    classes = [
        ("DJI", "Phantom4", "quadcopter"),
        ("DJI", "Mavic2", "quadcopter"),
        ("Parrot", "Bebop2", "fixed_wing"),
        ("Parrot", "ANAFI", "fixed_wing"),
        ("Autel", "EVO2", "hexacopter"),
        ("Skydio", "R1", "hexacopter"),
    ]
    rng = np.random.default_rng(0)
    for brand, model, drone_type in classes:
        for i in range(5):
            sig = processor.generate_synthetic_signal(
                drone_type=drone_type,
                duration=0.1,
                noise_level=rng.uniform(0.05, 0.25),
            )
            _, _, _, spec_norm = processor.process_signal(sig, (128, 128))
            handler.add_entry(spec_norm, brand, model, f"synthetic_{brand}_{model}_{i}.npy")


# ===========================================================================
# Tabs
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📡 Procesamiento STFT",
        "🗂️ Gestión del Dataset",
        "🤖 Entrenamiento CNN",
        "🔍 Identificación de Drones",
    ]
)

# ===========================================================================
# TAB 1 – Signal loading and STFT
# ===========================================================================
with tab1:
    st.header("Carga y Procesamiento de Señales RF")

    col_cfg, col_sig = st.columns([1, 2], gap="large")

    with col_cfg:
        st.subheader("⚙️ Configuración STFT")
        fs_mhz = st.number_input(
            "Frecuencia de Muestreo (MHz)", min_value=0.1, max_value=200.0, value=2.0, step=0.1
        )
        nperseg = st.select_slider(
            "Longitud de Ventana (muestras)", options=[64, 128, 256, 512, 1024], value=256
        )
        overlap_pct = st.slider("Solapamiento (%)", 0, 90, 75)
        nfft = st.selectbox("NFFT", [128, 256, 512, 1024, 2048], index=2)
        window_fn = st.selectbox("Función de Ventana", ["hann", "hamming", "blackman", "kaiser"])
        target_size = st.select_slider("Tamaño del Espectrograma", [64, 128, 256], value=128)

        if st.button("✅ Aplicar Configuración"):
            st.session_state.processor = SignalProcessor(
                fs=fs_mhz * 1e6,
                nperseg=nperseg,
                noverlap=int(nperseg * overlap_pct / 100),
                nfft=nfft,
                window=window_fn,
            )
            st.success("Configuración actualizada.")

    with col_sig:
        st.subheader("📂 Fuente de Señal")
        source = st.radio("Origen", ["Cargar Archivo", "Señal Sintética de Prueba"], horizontal=True)

        sig_array: np.ndarray | None = None
        sig_filename = ""

        if source == "Cargar Archivo":
            uploaded = st.file_uploader(
                "Seleccionar archivo de señal RF",
                type=["csv", "npy", "bin", "dat", "wav"],
                help="CSV (columnas I,Q) · NumPy (.npy) · IQ binario float32 (.bin/.dat) · WAV estéreo",
            )
            if uploaded:
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                try:
                    sig_array = st.session_state.processor.load_signal(uploaded, ext)
                    sig_filename = uploaded.name
                    st.success(f"✅ Señal cargada: **{len(sig_array):,}** muestras")
                except Exception as exc:
                    st.error(f"Error al cargar: {exc}")

        else:  # Synthetic
            drone_model = st.selectbox("Tipo de Dron", ["quadcopter", "fixed_wing", "hexacopter"])
            duration_ms = st.slider("Duración (ms)", 10, 1000, 100)
            noise_lvl = st.slider("Nivel de Ruido", 0.0, 1.0, 0.1, 0.01)
            if st.button("🎲 Generar Señal Sintética"):
                sig_array = st.session_state.processor.generate_synthetic_signal(
                    drone_type=drone_model,
                    duration=duration_ms / 1000.0,
                    noise_level=noise_lvl,
                )
                sig_filename = f"synthetic_{drone_model}_{_timestamp()}.npy"
                st.success(f"✅ Señal sintética generada: **{len(sig_array):,}** muestras")

        # Compute and display STFT
        if sig_array is not None:
            with st.spinner("Calculando STFT…"):
                f_arr, t_arr, spec_db, spec_norm = st.session_state.processor.process_signal(
                    sig_array, (target_size, target_size)
                )
            st.session_state.current_spec = {
                "spec_norm": spec_norm,
                "spec_db": spec_db,
                "f": f_arr,
                "t": t_arr,
                "filename": sig_filename,
            }
            fig = _plot_spectrograms(f_arr, t_arr, spec_db, spec_norm)
            st.pyplot(fig)
            plt.close(fig)

    # Add to dataset
    if st.session_state.current_spec is not None:
        st.markdown("---")
        st.subheader("🏷️ Etiquetar y Agregar al Dataset")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            brand_inp = st.text_input("Marca del Dron", placeholder="e.g. DJI, Parrot, Autel")
        with c2:
            type_inp = st.text_input("Tipo / Modelo", placeholder="e.g. Phantom4, Bebop2, EVO2")
        with c3:
            st.write("")
            st.write("")
            if st.button("➕ Agregar al Dataset", use_container_width=True):
                if brand_inp and type_inp:
                    cs = st.session_state.current_spec
                    st.session_state.dataset_handler.add_entry(
                        cs["spec_norm"], brand_inp, type_inp, cs["filename"]
                    )
                    n = st.session_state.dataset_handler.num_entries
                    st.success(f"✅ Agregado. Total en dataset: **{n}**")
                else:
                    st.error("Ingrese la marca y el tipo del dron.")

# ===========================================================================
# TAB 2 – Dataset management
# ===========================================================================
with tab2:
    st.header("Gestión del Dataset")

    col_load, col_export = st.columns(2, gap="large")

    with col_load:
        st.subheader("📥 Cargar Dataset Existente")
        ds_file = st.file_uploader(
            "Cargar archivo de dataset",
            type=["h5", "hdf5", "npz"],
            key="ds_upload",
            help="Formatos HDF5 (.h5 / .hdf5) o NPZ (.npz) generados por esta aplicación.",
        )
        if ds_file:
            ext = ds_file.name.rsplit(".", 1)[-1].lower()
            try:
                if ext in ("h5", "hdf5"):
                    st.session_state.dataset_handler.load_hdf5(ds_file)
                else:
                    st.session_state.dataset_handler.load_npz(ds_file)
                n = st.session_state.dataset_handler.num_entries
                st.success(f"✅ Dataset cargado: **{n}** entradas")
            except Exception as exc:
                st.error(f"Error al cargar dataset: {exc}")

    with col_export:
        st.subheader("💾 Exportar Dataset")
        handler = st.session_state.dataset_handler
        mc1, mc2 = st.columns(2)
        mc1.metric("Espectrogramas", handler.num_entries)
        mc2.metric("Clases", handler.num_classes)

        if handler.num_entries > 0:
            fmt = st.radio("Formato", ["HDF5", "NPZ"], horizontal=True)
            if fmt == "HDF5":
                buf = handler.export_hdf5()
                if buf:
                    st.download_button(
                        "💾 Descargar HDF5",
                        data=buf,
                        file_name=f"drone_rf_dataset_{_timestamp()}.h5",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
            else:
                buf = handler.export_npz()
                if buf:
                    st.download_button(
                        "💾 Descargar NPZ",
                        data=buf,
                        file_name=f"drone_rf_dataset_{_timestamp()}.npz",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
        else:
            st.info("Dataset vacío. Agregue señales en la pestaña **Procesamiento STFT**.")

    # Dataset content
    handler = st.session_state.dataset_handler
    if handler.num_entries > 0:
        st.markdown("---")
        st.subheader("📊 Contenido del Dataset")
        df = handler.get_dataframe()
        st.dataframe(df, use_container_width=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            brand_counts = df["brand"].value_counts()
            fig_b, ax_b = plt.subplots(figsize=(5, 3))
            brand_counts.plot(kind="bar", ax=ax_b, color="steelblue", edgecolor="white")
            ax_b.set_title("Distribución por Marca")
            ax_b.set_xlabel("Marca")
            ax_b.set_ylabel("Cantidad")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close(fig_b)

        with ch2:
            type_counts = df["type"].value_counts()
            fig_t, ax_t = plt.subplots(figsize=(5, 3))
            type_counts.plot(kind="pie", ax=ax_t, autopct="%1.1f%%", startangle=90)
            ax_t.set_title("Distribución por Tipo")
            ax_t.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

        # Sample spectrograms
        st.subheader("🖼️ Muestra de Espectrogramas")
        n_show = min(6, handler.num_entries)
        cols_sp = st.columns(n_show)
        for i, entry in enumerate(handler.entries[:n_show]):
            with cols_sp[i]:
                fig_s, ax_s = plt.subplots(figsize=(2.5, 2.5))
                ax_s.imshow(entry["spectrogram"], aspect="auto", origin="lower", cmap="viridis")
                ax_s.set_title(f"{entry['brand']}\n{entry['type']}", fontsize=8)
                ax_s.axis("off")
                st.pyplot(fig_s)
                plt.close(fig_s)

# ===========================================================================
# TAB 3 – CNN training
# ===========================================================================
with tab3:
    st.header("Entrenamiento del Modelo CNN")
    handler = st.session_state.dataset_handler

    MIN_SAMPLES = 10
    if handler.num_entries < MIN_SAMPLES:
        st.warning(
            f"⚠️ Se necesitan al menos **{MIN_SAMPLES}** muestras. "
            f"Actualmente: **{handler.num_entries}**."
        )
        if st.button("🎲 Generar Dataset Sintético de Prueba (30 muestras)"):
            with st.spinner("Generando…"):
                _generate_synthetic_dataset(st.session_state.processor, handler)
            st.success(f"✅ Dataset sintético listo: **{handler.num_entries}** muestras")
            st.rerun()
    else:
        col_cfg2, col_train = st.columns([1, 2], gap="large")

        with col_cfg2:
            st.subheader("⚙️ Configuración del Entrenamiento")
            n_splits = st.slider("Folds (K-Fold)", 2, 10, 5)
            epochs = st.slider("Épocas Máximas", 10, 200, 50)
            batch_size = st.select_slider("Batch Size", [8, 16, 32, 64, 128], value=32)

            st.markdown("---")
            X_info, y_info, labels_info = handler.get_arrays()
            st.subheader("Información del Modelo")
            st.write(f"**Forma de entrada:** {X_info.shape[1:]} + canal")
            st.write(f"**Clases ({len(labels_info)}):**")
            for idx, lbl in enumerate(labels_info):
                st.write(f"  `{idx}` → {lbl}")
            st.code(
                "Conv2D(32)×2 + BN + MaxPool\n"
                "Conv2D(64)×2 + BN + MaxPool\n"
                "Conv2D(128)×2 + BN + MaxPool\n"
                "GlobalAvgPool\n"
                "Dense(256) + Dropout(0.5)\n"
                f"Dense({len(labels_info)}, softmax)",
                language="text",
            )

        with col_train:
            st.subheader("Validación Cruzada y Entrenamiento")
            btn_cv, btn_final = st.columns(2)
            run_cv = btn_cv.button("🔄 Validación Cruzada", type="primary", use_container_width=True)
            run_final = btn_final.button("🚀 Entrenar Modelo Final", use_container_width=True)

            # ---- Cross-validation ----
            if run_cv:
                X, y, labels = handler.get_arrays()
                n_classes = len(labels)

                prog = st.progress(0.0)
                status = st.empty()

                def _progress(fold, total):
                    prog.progress(fold / total)
                    status.text(f"Fold {fold}/{total}…")

                trainer = ModelTrainer(n_splits=n_splits, epochs=epochs, batch_size=batch_size)

                def _builder():
                    return build_cnn((*X.shape[1:], 1), n_classes)

                try:
                    with st.spinner("Ejecutando validación cruzada…"):
                        cv_res = trainer.cross_validate(_builder, X, y, labels, _progress)

                    st.session_state.cv_results = cv_res
                    st.session_state.label_map = labels
                    prog.progress(1.0)
                    status.text("✅ Completado")

                    summary = trainer.get_cv_summary()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precisión Media", f"{summary['mean_accuracy']:.3f}")
                    m2.metric("Desv. Estándar", f"±{summary['std_accuracy']:.3f}")
                    m3.metric("Pérdida Media", f"{summary['mean_loss']:.3f}")

                    fold_df = trainer.get_cv_dataframe()
                    fig_cv, ax_cv = plt.subplots(1, 2, figsize=(11, 3.5))
                    ax_cv[0].bar(fold_df["Fold"], fold_df["Accuracy"], color="steelblue")
                    ax_cv[0].axhline(
                        summary["mean_accuracy"], color="red", linestyle="--",
                        label=f"Media={summary['mean_accuracy']:.3f}"
                    )
                    ax_cv[0].set_ylim(0, 1)
                    ax_cv[0].set_xlabel("Fold")
                    ax_cv[0].set_ylabel("Precisión")
                    ax_cv[0].set_title("Precisión por Fold")
                    ax_cv[0].legend()

                    ax_cv[1].bar(fold_df["Fold"], fold_df["Loss"], color="coral")
                    ax_cv[1].axhline(
                        summary["mean_loss"], color="red", linestyle="--",
                        label=f"Media={summary['mean_loss']:.3f}"
                    )
                    ax_cv[1].set_xlabel("Fold")
                    ax_cv[1].set_ylabel("Pérdida")
                    ax_cv[1].set_title("Pérdida por Fold")
                    ax_cv[1].legend()
                    plt.tight_layout()
                    st.pyplot(fig_cv)
                    plt.close(fig_cv)

                    # Confusion matrix – last fold
                    last = cv_res[-1]
                    cm = last["confusion_matrix"]
                    fig_cm, ax_cm = plt.subplots(figsize=(max(5, n_classes), max(4, n_classes - 1)))
                    im_cm = ax_cm.imshow(cm, cmap="Blues")
                    plt.colorbar(im_cm, ax=ax_cm)
                    ax_cm.set_xticks(range(n_classes))
                    ax_cm.set_yticks(range(n_classes))
                    ax_cm.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                    ax_cm.set_yticklabels(labels, fontsize=8)
                    ax_cm.set_xlabel("Predicción")
                    ax_cm.set_ylabel("Real")
                    ax_cm.set_title("Matriz de Confusión (Último Fold)")
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax_cm.text(
                                j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8,
                            )
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                except Exception as exc:
                    st.error(f"Error en validación cruzada: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

            elif st.session_state.cv_results is not None:
                st.info("Resultados de CV disponibles. Entrene el modelo final cuando esté listo.")

            # ---- Final model training ----
            if run_final:
                X, y, labels = handler.get_arrays()
                n_classes = len(labels)
                model = build_cnn((*X.shape[1:], 1), n_classes)
                trainer_f = ModelTrainer(epochs=epochs, batch_size=batch_size)

                try:
                    with st.spinner("Entrenando modelo final…"):
                        hist = trainer_f.train_final_model(model, X, y)

                    st.session_state.trained_model = model
                    st.session_state.label_map = labels
                    st.success("✅ Modelo final entrenado exitosamente")

                    fig_h, ax_h = plt.subplots(1, 2, figsize=(11, 3.5))
                    ax_h[0].plot(hist.history["accuracy"], label="Train")
                    if "val_accuracy" in hist.history:
                        ax_h[0].plot(hist.history["val_accuracy"], label="Val")
                    ax_h[0].set_ylim(0, 1)
                    ax_h[0].set_xlabel("Época")
                    ax_h[0].set_ylabel("Precisión")
                    ax_h[0].set_title("Precisión")
                    ax_h[0].legend()

                    ax_h[1].plot(hist.history["loss"], label="Train")
                    if "val_loss" in hist.history:
                        ax_h[1].plot(hist.history["val_loss"], label="Val")
                    ax_h[1].set_xlabel("Época")
                    ax_h[1].set_ylabel("Pérdida")
                    ax_h[1].set_title("Pérdida")
                    ax_h[1].legend()
                    plt.tight_layout()
                    st.pyplot(fig_h)
                    plt.close(fig_h)

                    model_bytes = save_model_to_bytes(model)
                    st.download_button(
                        "💾 Descargar Modelo (.h5)",
                        data=model_bytes,
                        file_name=f"drone_cnn_{_timestamp()}.h5",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"Error al entrenar: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

# ===========================================================================
# TAB 4 – Identification
# ===========================================================================
with tab4:
    st.header("Identificación de Drones por RF")

    # Load model if not trained yet
    if st.session_state.trained_model is None:
        st.warning("⚠️ Sin modelo entrenado. Entrene uno en la pestaña **Entrenamiento CNN**.")
        st.subheader("O cargar un modelo existente (.h5)")
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            model_file = st.file_uploader("Modelo Keras (.h5)", type=["h5"], key="model_upload")
        with col_ml2:
            labels_text = st.text_input(
                "Etiquetas (separadas por coma)",
                placeholder="DJI_Phantom4, Parrot_Bebop2, Autel_EVO2",
            )
        if model_file and labels_text:
            try:
                loaded_model = load_model_from_bytes(model_file.read())
                st.session_state.trained_model = loaded_model
                st.session_state.label_map = [l.strip() for l in labels_text.split(",")]
                st.success("✅ Modelo cargado.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error al cargar modelo: {exc}")
    else:
        label_map: list[str] = st.session_state.label_map
        model = st.session_state.trained_model

        st.success(
            f"✅ Modelo listo · {len(label_map)} clases: "
            + ", ".join(f"`{l}`" for l in label_map)
        )
        st.markdown("---")

        col_inp, col_res = st.columns([1, 1], gap="large")

        with col_inp:
            st.subheader("Señal a Identificar")
            id_src = st.radio("Fuente", ["Cargar Archivo", "Señal Sintética"], horizontal=True, key="id_src")
            sig_id: np.ndarray | None = None

            if id_src == "Cargar Archivo":
                id_file = st.file_uploader(
                    "Archivo de señal RF",
                    type=["csv", "npy", "bin", "dat", "wav"],
                    key="id_file",
                )
                if id_file:
                    ext = id_file.name.rsplit(".", 1)[-1].lower()
                    try:
                        sig_id = st.session_state.processor.load_signal(id_file, ext)
                        st.info(f"Muestras: **{len(sig_id):,}**")
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            else:
                id_drone = st.selectbox("Tipo Simulado", ["quadcopter", "fixed_wing", "hexacopter"], key="id_drone")
                if st.button("🎲 Generar", key="id_gen"):
                    sig_id = st.session_state.processor.generate_synthetic_signal(
                        drone_type=id_drone, duration=0.1, noise_level=0.1
                    )

            if sig_id is not None:
                target_h = int(model.input_shape[1])
                target_w = int(model.input_shape[2])
                _, _, spec_db_id, spec_norm_id = st.session_state.processor.process_signal(
                    sig_id, (target_h, target_w)
                )
                fig_id = _plot_spectrograms(
                    np.linspace(0, 1, spec_norm_id.shape[0]),
                    np.linspace(0, 1, spec_norm_id.shape[1]),
                    spec_norm_id,
                    spec_norm_id,
                    title_prefix="Entrada – ",
                )
                st.pyplot(fig_id)
                plt.close(fig_id)

                if st.button("🔍 Identificar Dron", type="primary", use_container_width=True):
                    X_in = spec_norm_id[np.newaxis, ..., np.newaxis]
                    probs = model.predict(X_in, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = label_map[pred_idx]
                    parts = pred_label.split("_", 1)
                    pred_brand = parts[0]
                    pred_type = parts[1] if len(parts) > 1 else "Desconocido"

                    with col_res:
                        st.subheader("📋 Resultado")
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Marca", pred_brand)
                        r2.metric("Modelo/Tipo", pred_type)
                        r3.metric("Confianza", f"{probs[pred_idx]:.1%}")

                        prob_df = (
                            pd.DataFrame({"Clase": label_map, "Probabilidad": probs})
                            .sort_values("Probabilidad", ascending=True)
                        )
                        bar_colors = [
                            "#d62728" if c == pred_label else "#1f77b4"
                            for c in prob_df["Clase"]
                        ]
                        fig_bar, ax_bar = plt.subplots(
                            figsize=(7, max(3, len(label_map) * 0.55))
                        )
                        ax_bar.barh(prob_df["Clase"], prob_df["Probabilidad"], color=bar_colors)
                        ax_bar.set_xlabel("Probabilidad")
                        ax_bar.set_title("Distribución de Probabilidades")
                        ax_bar.set_xlim(0, 1)
                        plt.tight_layout()
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)

# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.title("🛸 DronRForensic")
    st.markdown("---")

    st.subheader("Estado")
    h = st.session_state.dataset_handler
    st.metric("Espectrogramas en Dataset", h.num_entries)
    st.metric("Clases", h.num_classes)
    st.metric(
        "Modelo",
        "✅ Entrenado" if st.session_state.trained_model is not None else "❌ No entrenado",
    )

    st.markdown("---")
    st.subheader("Flujo de Trabajo")
    st.markdown(
        """
1. **📡 STFT** – Carga señales RF y genera espectrogramas.
2. **🏷️ Etiqueta** – Asigna marca y modelo a cada señal.
3. **💾 Exporta** – Descarga el dataset para reusar.
4. **🤖 Entrena** – Validación cruzada K-fold + CNN.
5. **🔍 Identifica** – Clasifica drones desconocidos.
        """
    )

    st.markdown("---")
    if h.num_entries > 0 and st.button("🗑️ Limpiar Dataset", use_container_width=True):
        st.session_state.dataset_handler.clear()
        st.session_state.cv_results = None
        st.session_state.trained_model = None
        st.session_state.label_map = None
        st.rerun()

    st.caption("© 2024 DronRForensic | DTL-DA")
