"""
DroneRF Analyzer - Aplicación Principal
Plataforma para recopilar señales RF de drones, generar espectrogramas via STFT,
etiquetar datos y entrenar modelos CNN para identificación de huellas RF.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from modules.signal_processing import cargar_senal, generar_espectrograma, espectrograma_a_imagen, procesar_lote
from modules.labeling import GestorEtiquetas, MARCAS_DRONES, TIPOS_DRONES, preparar_dataset, crear_zip_dataset
from modules.cnn_model import entrenar_con_validacion_cruzada, generar_reporte_texto, exportar_modelo

# ========================
# CONFIGURACIÓN DE LA APP
# ========================
st.set_page_config(
    page_title="DroneRF Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.8; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# ESTADO DE SESIÓN
# ========================
if 'resultados' not in st.session_state:
    st.session_state.resultados = []
if 'gestor_etiquetas' not in st.session_state:
    st.session_state.gestor_etiquetas = GestorEtiquetas()
if 'resultados_entrenamiento' not in st.session_state:
    st.session_state.resultados_entrenamiento = None
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = None

gestor = st.session_state.gestor_etiquetas

# ========================
# HEADER
# ========================
st.markdown("""
<div class="main-header">
    <h1>📡 DroneRF Analyzer</h1>
    <p style="color: #888; font-size: 1.1rem;">
        Análisis de huellas RF de drones · STFT · Espectrogramas · CNN
    </p>
</div>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("### ⚙️ Parámetros STFT")
    
    nperseg = st.select_slider(
        "Ventana (nperseg)",
        options=[64, 128, 256, 512, 1024, 2048],
        value=256,
        help="Número de muestras por segmento. Mayor = mejor resolución en frecuencia."
    )
    
    solapamiento = st.slider(
        "Solapamiento (%)",
        min_value=25, max_value=90, value=50, step=5,
        help="Porcentaje de solapamiento entre ventanas."
    )
    noverlap = int(nperseg * solapamiento / 100)
    
    ventana = st.selectbox(
        "Tipo de ventana",
        ['hann', 'hamming', 'blackman', 'kaiser', 'bartlett', 'flattop'],
        index=0,
        help="Función de ventana para la STFT."
    )
    
    fs_manual = st.number_input(
        "Frecuencia muestreo (Hz)",
        min_value=1000, max_value=100000000,
        value=2400000, step=100000,
        help="Frecuencia de muestreo. Déjalo en 2.4MHz para señales de drones WiFi/RC."
    )
    
    tamano_img = st.select_slider(
        "Tamaño imagen CNN",
        options=[64, 128, 224, 256],
        value=128,
        help="Resolución de la imagen del espectrograma para la CNN."
    )

    st.markdown("---")
    st.markdown("### 📊 Resumen")
    st.metric("Señales cargadas", len(st.session_state.resultados))
    st.metric("Etiquetadas", gestor.total_etiquetados())

# ========================
# PESTAÑAS PRINCIPALES
# ========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 1. Cargar y Transformar",
    "🏷️ 2. Etiquetar",
    "📦 3. Dataset",
    "🧠 4. Entrenar CNN"
])

# ========================
# TAB 1: CARGAR Y TRANSFORMAR
# ========================
with tab1:
    st.markdown("## 📂 Cargar Señales RF y Generar Espectrogramas")
    st.markdown("Sube archivos con señales RF de drones. Formatos: **CSV, NPY, NPZ, WAV, JSON, TXT, BIN/RAW/IQ, MAT**")
    
    archivos = st.file_uploader(
        "Selecciona archivos de señales RF",
        accept_multiple_files=True,
        type=['csv', 'npy', 'npz', 'wav', 'json', 'txt', 'bin', 'raw', 'iq', 'mat'],
        key="uploader_signals"
    )
    
    if archivos and st.button("🔄 Procesar Señales (STFT)", type="primary", use_container_width=True):
        with st.spinner("Procesando señales con STFT..."):
            lista_archivos = [(f.name, f.read()) for f in archivos]
            
            resultados = procesar_lote(
                lista_archivos,
                fs_override=fs_manual,
                nperseg=nperseg,
                noverlap=noverlap,
                ventana=ventana,
                tamano_img=(tamano_img, tamano_img)
            )
            
            st.session_state.resultados = resultados
            
            exitosos = sum(1 for r in resultados if not r.get('error'))
            errores = sum(1 for r in resultados if r.get('error'))
            
            if exitosos > 0:
                st.success(f"✅ {exitosos} señales procesadas exitosamente.")
            if errores > 0:
                st.warning(f"⚠️ {errores} archivos con errores.")
    
    # Función auxiliar para submuestrear espectrograma
    def _submuestrear_espectrograma(r, max_cols=2000):
        t_p = r['t']
        f_p = r['f'] / 1e6
        s_p = r['Sxx_db']
        if s_p.shape[1] > max_cols:
            step = s_p.shape[1] // max_cols
            s_p = s_p[:, ::step]
            t_p = t_p[::step]
        return t_p, f_p, s_p

    # Dialog para ver espectrograma en grande
    @st.dialog("🔍 Espectrograma Detallado", width="large")
    def ver_espectrograma(idx):
        resultados_ok = [r for r in st.session_state.resultados if not r.get('error')]
        r = resultados_ok[idx]
        st.markdown(f"### {r['nombre']}")
        st.markdown(f"**Frecuencia de muestreo:** {r['fs']/1e6:.2f} MHz · **Muestras:** {len(r['senal']):,}")

        # Señal temporal
        fig_signal, ax_s = plt.subplots(figsize=(14, 3))
        senal_plot = np.real(r['senal'][:min(5000, len(r['senal']))])
        ax_s.plot(senal_plot, linewidth=0.4, color='#667eea')
        ax_s.set_title('Señal Temporal (primeras 5000 muestras)', fontsize=12)
        ax_s.set_xlabel('Muestras')
        ax_s.set_ylabel('Amplitud')
        ax_s.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_signal)
        plt.close(fig_signal)

        # Espectrograma grande
        t_p, f_p, s_p = _submuestrear_espectrograma(r, max_cols=3000)
        fig_spec, ax_sp = plt.subplots(figsize=(14, 6))
        im = ax_sp.pcolormesh(t_p, f_p, s_p, shading='gouraud', cmap='viridis')
        ax_sp.set_title('Espectrograma STFT', fontsize=14)
        ax_sp.set_xlabel('Tiempo (s)', fontsize=11)
        ax_sp.set_ylabel('Frecuencia (MHz)', fontsize=11)
        plt.colorbar(im, ax=ax_sp, label='Potencia (dB)')
        plt.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        # Imagen CNN preview
        st.markdown("**Vista previa imagen CNN:**")
        fig_cnn, ax_c = plt.subplots(figsize=(4, 4))
        ax_c.imshow(r['imagen_cnn'][:, :, 0], cmap='viridis', aspect='auto')
        ax_c.set_title(f"Imagen CNN ({r['imagen_cnn'].shape[0]}x{r['imagen_cnn'].shape[1]})")
        ax_c.axis('off')
        plt.tight_layout()
        st.pyplot(fig_cnn)
        plt.close(fig_cnn)

    # Mostrar espectrogramas generados
    if st.session_state.resultados:
        st.markdown("### 🎨 Espectrogramas Generados")
        
        resultados_ok = [r for r in st.session_state.resultados if not r.get('error')]
        
        for i in range(0, len(resultados_ok), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(resultados_ok):
                    break
                r = resultados_ok[idx]
                
                with col:
                    # Miniatura del espectrograma
                    t_p, f_p, s_p = _submuestrear_espectrograma(r, max_cols=500)
                    fig_mini, ax_mini = plt.subplots(figsize=(4, 2.5))
                    ax_mini.pcolormesh(t_p, f_p, s_p, shading='gouraud', cmap='viridis')
                    ax_mini.set_title(r['nombre'], fontsize=8)
                    ax_mini.set_xlabel('t (s)', fontsize=7)
                    ax_mini.set_ylabel('f (MHz)', fontsize=7)
                    ax_mini.tick_params(labelsize=6)
                    plt.tight_layout()
                    st.pyplot(fig_mini)
                    plt.close(fig_mini)

                    st.caption(f"fs: {r['fs']/1e6:.2f} MHz · {len(r['senal']):,} muestras")
                    if st.button(f"🔍 Ver detalle", key=f"ver_{idx}", use_container_width=True):
                        ver_espectrograma(idx)
        
        # Errores
        errores = [r for r in st.session_state.resultados if r.get('error')]
        if errores:
            with st.expander(f"❌ {len(errores)} archivos con errores"):
                for r in errores:
                    st.error(f"**{r['nombre']}**: {r['error']}")

# ========================
# TAB 2: ETIQUETAR
# ========================
with tab2:
    st.markdown("## 🏷️ Etiquetar Espectrogramas")
    
    resultados_ok = [r for r in st.session_state.resultados if not r.get('error')]
    
    if not resultados_ok:
        st.info("⬆️ Primero carga y procesa señales RF en la pestaña anterior.")
    else:
        # Etiquetado masivo
        st.markdown("### 🚀 Etiquetado Rápido (todos a la vez)")
        c1, c2, c3 = st.columns(3)
        with c1:
            marca_masiva = st.selectbox("Marca (todos)", MARCAS_DRONES, key="marca_masiva")
        with c2:
            tipo_masivo = st.selectbox("Tipo (todos)", TIPOS_DRONES, key="tipo_masivo")
        with c3:
            huella_masiva = st.text_input("ID Huella RF (todos)", key="huella_masiva", placeholder="Ej: DJI-MAVIC3-001")
        
        if st.button("🏷️ Aplicar a Todos", type="primary"):
            for r in resultados_ok:
                gestor.etiquetar(r['nombre'], marca_masiva, tipo_masivo, huella_masiva)
            st.success(f"✅ {len(resultados_ok)} espectrogramas etiquetados.")
            st.rerun()
        
        st.markdown("---")
        
        # Etiquetado individual
        st.markdown("### 🔧 Etiquetado Individual")
        
        for r in resultados_ok:
            etiqueta_actual = gestor.obtener_etiqueta(r['nombre'])
            estado = "✅" if etiqueta_actual else "⬜"
            
            with st.expander(f"{estado} {r['nombre']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Mini espectrograma
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.pcolormesh(r['t'], r['f'] / 1e6, r['Sxx_db'],
                                  shading='gouraud', cmap='viridis')
                    ax.set_xlabel('Tiempo (s)', fontsize=8)
                    ax.set_ylabel('Freq (MHz)', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    marca_default = MARCAS_DRONES.index(etiqueta_actual['marca']) if etiqueta_actual and etiqueta_actual['marca'] in MARCAS_DRONES else 0
                    tipo_default = TIPOS_DRONES.index(etiqueta_actual['tipo']) if etiqueta_actual and etiqueta_actual['tipo'] in TIPOS_DRONES else 0
                    
                    marca = st.selectbox(
                        "Marca", MARCAS_DRONES,
                        index=marca_default,
                        key=f"marca_{r['nombre']}"
                    )
                    tipo = st.selectbox(
                        "Tipo", TIPOS_DRONES,
                        index=tipo_default,
                        key=f"tipo_{r['nombre']}"
                    )
                    huella = st.text_input(
                        "ID Huella RF",
                        value=etiqueta_actual.get('huella_id', '') if etiqueta_actual else '',
                        key=f"huella_{r['nombre']}",
                        placeholder="Ej: DJI-MAVIC3-001"
                    )
                    notas = st.text_input(
                        "Notas",
                        value=etiqueta_actual.get('notas', '') if etiqueta_actual else '',
                        key=f"notas_{r['nombre']}"
                    )
                    
                    if st.button("💾 Guardar", key=f"save_{r['nombre']}"):
                        gestor.etiquetar(r['nombre'], marca, tipo, huella, notas)
                        st.success("Etiqueta guardada.")
                        st.rerun()
        
        # Importar/Exportar etiquetas
        st.markdown("---")
        st.markdown("### 📋 Importar/Exportar Etiquetas")
        c1, c2 = st.columns(2)
        
        with c1:
            if gestor.total_etiquetados() > 0:
                st.download_button(
                    "📥 Descargar Etiquetas (CSV)",
                    gestor.exportar_csv(),
                    "etiquetas_dronerf.csv",
                    "text/csv"
                )
                st.download_button(
                    "📥 Descargar Etiquetas (JSON)",
                    gestor.exportar_json(),
                    "etiquetas_dronerf.json",
                    "application/json"
                )
        
        with c2:
            archivo_etiquetas = st.file_uploader(
                "📤 Importar etiquetas", type=['csv', 'json'],
                key="import_labels"
            )
            if archivo_etiquetas:
                contenido = archivo_etiquetas.read().decode('utf-8')
                if archivo_etiquetas.name.endswith('.json'):
                    gestor.importar_json(contenido)
                else:
                    gestor.importar_csv(contenido)
                st.success(f"Etiquetas importadas. Total: {gestor.total_etiquetados()}")
                st.rerun()

# ========================
# TAB 3: DATASET
# ========================
with tab3:
    st.markdown("## 📦 Preparar y Descargar Dataset")
    
    resultados_ok = [r for r in st.session_state.resultados if not r.get('error')]
    etiquetados = [r for r in resultados_ok if gestor.obtener_etiqueta(r['nombre'])]
    
    if not etiquetados:
        st.info("⬆️ Primero carga señales y etiquétalas en las pestañas anteriores.")
    else:
        # Resumen del dataset
        st.markdown("### 📊 Resumen del Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h3>Total Muestras</h3><h1>{len(etiquetados)}</h1>
            </div>""", unsafe_allow_html=True)
        with col2:
            marcas = gestor.obtener_clases_marca()
            st.markdown(f"""<div class="metric-card">
                <h3>Marcas Únicas</h3><h1>{len(marcas)}</h1>
            </div>""", unsafe_allow_html=True)
        with col3:
            tipos = gestor.obtener_clases_tipo()
            st.markdown(f"""<div class="metric-card">
                <h3>Tipos Únicos</h3><h1>{len(tipos)}</h1>
            </div>""", unsafe_allow_html=True)
        
        # Distribución
        st.markdown("### 📈 Distribución de Clases")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Por Marca:**")
            conteo_marcas = {}
            for r in etiquetados:
                et = gestor.obtener_etiqueta(r['nombre'])
                conteo_marcas[et['marca']] = conteo_marcas.get(et['marca'], 0) + 1
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(list(conteo_marcas.keys()), list(conteo_marcas.values()),
                          color='#667eea', edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Cantidad')
            ax.set_title('Distribución por Marca')
            for bar, val in zip(bars, conteo_marcas.values()):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                       str(val), va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with c2:
            st.markdown("**Por Tipo:**")
            conteo_tipos = {}
            for r in etiquetados:
                et = gestor.obtener_etiqueta(r['nombre'])
                conteo_tipos[et['tipo']] = conteo_tipos.get(et['tipo'], 0) + 1
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(list(conteo_tipos.keys()), list(conteo_tipos.values()),
                          color='#764ba2', edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Cantidad')
            ax.set_title('Distribución por Tipo')
            for bar, val in zip(bars, conteo_tipos.values()):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                       str(val), va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Descarga
        st.markdown("### 💾 Descargar Dataset")
        target_descarga = st.radio(
            "Clasificar por:", ["marca", "tipo"],
            horizontal=True, key="target_descarga"
        )
        
        if st.button("📦 Generar ZIP del Dataset", type="primary", use_container_width=True):
            with st.spinner("Empaquetando dataset..."):
                zip_buffer = crear_zip_dataset(
                    st.session_state.resultados, gestor, target=target_descarga
                )
                st.download_button(
                    "⬇️ Descargar Dataset (ZIP)",
                    zip_buffer.getvalue(),
                    f"dronerf_dataset_{target_descarga}.zip",
                    "application/zip",
                    use_container_width=True
                )
                st.success("Dataset listo para descargar.")

# ========================
# TAB 4: ENTRENAR CNN
# ========================
with tab4:
    st.markdown("## 🧠 Entrenar modelo CNN con Validación Cruzada")
    
    resultados_ok = [r for r in st.session_state.resultados if not r.get('error')]
    etiquetados = [r for r in resultados_ok if gestor.obtener_etiqueta(r['nombre'])]
    
    if len(etiquetados) < 5:
        st.info(f"⬆️ Necesitas al menos 5 muestras etiquetadas para entrenar. Tienes: {len(etiquetados)}")
    else:
        st.markdown("### ⚙️ Configuración de Entrenamiento")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            target = st.selectbox("Clasificar por", ["marca", "tipo"], key="target_train")
        with col2:
            n_folds = st.selectbox("K-Folds", [3, 5, 10], index=1, key="nfolds")
        with col3:
            epochs = st.selectbox("Épocas máx.", [20, 50, 100, 200], index=1, key="epochs")
        with col4:
            batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2, key="batch")
        
        # Verificar que hay suficientes clases
        X, y, clases, nombres = preparar_dataset(
            st.session_state.resultados, gestor, target=target
        )
        
        if X is None:
            st.warning("No hay datos suficientes con las etiquetas seleccionadas.")
        else:
            st.info(f"📊 Dataset: **{len(X)}** muestras, **{len(clases)}** clases: {', '.join(clases)}")
            
            # Verificar mínimo por clase para k-fold
            from collections import Counter
            conteo = Counter(y.tolist())
            min_muestras = min(conteo.values())
            
            if min_muestras < n_folds:
                st.warning(
                    f"⚠️ La clase con menos muestras tiene {min_muestras}. "
                    f"Reduce K-Folds a {min_muestras} o menos, o agrega más datos."
                )
                n_folds = min(n_folds, min_muestras)
            
            if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
                st.markdown("---")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def callback_progreso(fold_actual, total_folds, resultado_fold):
                    progress = fold_actual / total_folds
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f"**Fold {fold_actual}/{total_folds}** — "
                        f"Accuracy: {resultado_fold['accuracy']:.4f} | "
                        f"F1: {resultado_fold['f1_score']:.4f}"
                    )
                
                with st.spinner("Entrenando CNN... Esto puede tardar varios minutos."):
                    resultados_cv, modelo = entrenar_con_validacion_cruzada(
                        X, y, clases,
                        n_folds=n_folds,
                        epochs=epochs,
                        batch_size=batch_size,
                        callback_progreso=callback_progreso
                    )
                
                st.session_state.resultados_entrenamiento = resultados_cv
                st.session_state.modelo_entrenado = modelo
                
                progress_bar.progress(1.0)
                status_text.success("✅ Entrenamiento completado!")
        
        # Mostrar resultados
        if st.session_state.resultados_entrenamiento:
            res = st.session_state.resultados_entrenamiento
            
            st.markdown("---")
            st.markdown("### 📊 Resultados de Validación Cruzada")
            
            # Métricas principales
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Accuracy", f"{res['accuracy_promedio']:.4f}",
                          delta=f"±{res['accuracy_std']:.4f}")
            with c2:
                st.metric("Precision", f"{res['precision_promedio']:.4f}")
            with c3:
                st.metric("Recall", f"{res['recall_promedio']:.4f}")
            with c4:
                st.metric("F1-Score", f"{res['f1_promedio']:.4f}")
            
            # Gráficas
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Accuracy por Fold**")
                fig, ax = plt.subplots(figsize=(6, 4))
                folds = [f"Fold {f['fold']}" for f in res['folds']]
                accs = [f['accuracy'] for f in res['folds']]
                bars = ax.bar(folds, accs, color='#667eea', edgecolor='white')
                ax.axhline(y=res['accuracy_promedio'], color='red',
                          linestyle='--', label=f"Promedio: {res['accuracy_promedio']:.4f}")
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('Accuracy')
                ax.legend()
                for bar, val in zip(bars, accs):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            with c2:
                st.markdown("**Matriz de Confusión**")
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = np.array(res['matriz_confusion'])
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_xticks(range(len(res['clases'])))
                ax.set_yticks(range(len(res['clases'])))
                ax.set_xticklabels(res['clases'], rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(res['clases'], fontsize=8)
                ax.set_xlabel('Predicho')
                ax.set_ylabel('Real')
                for i in range(len(cm)):
                    for j in range(len(cm[0])):
                        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                               fontsize=10, color='white' if cm[i][j] > cm.max()/2 else 'black')
                plt.colorbar(im)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Curvas de entrenamiento
            if res.get('historiales'):
                st.markdown("**Curvas de Entrenamiento (último fold)**")
                hist = res['historiales'][-1]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(hist['loss'], label='Train Loss', color='#667eea')
                ax1.plot(hist['val_loss'], label='Val Loss', color='#e74c3c')
                ax1.set_xlabel('Época')
                ax1.set_ylabel('Loss')
                ax1.set_title('Pérdida')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(hist['accuracy'], label='Train Acc', color='#667eea')
                ax2.plot(hist['val_accuracy'], label='Val Acc', color='#2ecc71')
                ax2.set_xlabel('Época')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Precisión')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Descargas
            st.markdown("### 💾 Descargar Resultados")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                reporte_texto = generar_reporte_texto(res)
                st.download_button(
                    "📄 Reporte (TXT)",
                    reporte_texto,
                    "reporte_validacion_cruzada.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with c2:
                import json
                st.download_button(
                    "📊 Resultados (JSON)",
                    json.dumps(res, indent=2, ensure_ascii=False),
                    "resultados_cv.json",
                    "application/json",
                    use_container_width=True
                )
            
            with c3:
                if st.session_state.modelo_entrenado:
                    if st.button("🧠 Exportar Modelo", use_container_width=True):
                        with st.spinner("Exportando modelo..."):
                            model_zip = exportar_modelo(
                                st.session_state.modelo_entrenado,
                                res['clases']
                            )
                            st.download_button(
                                "⬇️ Descargar Modelo (ZIP)",
                                model_zip.getvalue(),
                                "modelo_dronerf_cnn.zip",
                                "application/zip",
                                use_container_width=True
                            )

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem 0;">
    DroneRF Analyzer v1.0 · STFT + CNN para identificación de huellas RF de drones<br>
    Transformada de Fourier de Ventana Corta · Validación Cruzada K-Fold · Redes Neuronales Convolucionales
</div>
""", unsafe_allow_html=True)
