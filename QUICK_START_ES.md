# Guía de Inicio Rápido - YOLO Cell Count GUI

## 🎯 Objetivo
Esta guía te ayudará a usar la interfaz gráfica para detectar y contar células usando YOLO.

## 📋 Prerrequisitos

Antes de empezar, asegúrate de tener instalado:
1. Python 3.7 o superior
2. Las dependencias del proyecto (ver sección de instalación en README.md)

## 🚀 Paso a Paso

### 1. Instalar dependencias
```bash
cd YOLO-cell-count
pip install -e .
```

### 2. Ejecutar la interfaz gráfica
```bash
python LiveApp.py
```

### 3. Usar la GUI

Una vez abierta la ventana, verás:

#### 🎥 Sección: Cámaras Detectadas
- Muestra automáticamente las cámaras web conectadas a tu sistema
- Si no detecta cámaras, verás un mensaje en rojo

#### ⚙️ Sección: Parámetros de Inferencia
Configura los siguientes parámetros según tus necesidades:

- **Inference Size** (320-1280): Tamaño de las imágenes para inferencia
  - Valor por defecto: 640
  - Valores más altos = mayor precisión pero más lento

- **Confidence Threshold** (0.0-1.0): Umbral de confianza mínimo
  - Valor por defecto: 0.5
  - Valores más altos = menos detecciones pero más confiables

- **IOU Threshold** (0.0-1.0): Umbral de intersección sobre unión
  - Valor por defecto: 0.4
  - Controla la supresión de detecciones duplicadas

- **Box Thickness** (0.1-10.0): Grosor de las cajas de detección
  - Valor por defecto: 0.5

- **Font Size** (0.1-5.0): Tamaño de texto en las anotaciones
  - Valor por defecto: 0.6

- **Webcam ID** (0-10): ID de la cámara a usar
  - Valor por defecto: 0
  - Usa el número mostrado en "Cámaras Detectadas"

#### 🎮 Botones

- **Restablecer Valores Predeterminados**: Vuelve todos los parámetros a sus valores originales
- **Ejecutar Script**: Inicia la detección con los parámetros configurados

### 4. Usar la ventana de detección

Una vez que hagas clic en "Ejecutar Script", se abrirá una nueva ventana mostrando:
- El video de tu cámara en tiempo real
- Las detecciones de células (cuando actives la inferencia)
- El conteo de células detectadas
- Uso de CPU y RAM

#### ⌨️ Controles del teclado:

| Tecla | Función |
|-------|---------|
| `q` | Salir |
| `i` | Activar/Desactivar inferencia |
| `m` | Alternar colores multicolor en las cajas |
| `c` | Aumentar contraste |
| `v` | Disminuir contraste |
| `b` | Aumentar brillo |
| `n` | Disminuir brillo |
| `p` | Tomar captura de pantalla |

## ❓ Preguntas Frecuentes

**P: ¿Necesito un modelo YOLO?**
R: Sí, el código busca un archivo llamado `RCCD-8n640.onnx` en el directorio actual. Asegúrate de tenerlo antes de ejecutar.

**P: ¿Puedo usar múltiples cámaras?**
R: Sí, selecciona el ID de la cámara deseada en el campo "Webcam ID" de la GUI.

**P: ¿Qué hago si no se detectan cámaras?**
R: Verifica que tu cámara esté conectada y funcionando. En Windows, asegúrate de que no esté siendo usada por otra aplicación.

## 💡 Consejos

1. **Primera vez**: Comienza con los valores predeterminados
2. **Detección lenta**: Reduce el "Inference Size"
3. **Pocas detecciones**: Reduce el "Confidence Threshold"
4. **Muchas detecciones falsas**: Aumenta el "Confidence Threshold"
5. **Cajas duplicadas**: Ajusta el "IOU Threshold"

## 🐛 Solución de Problemas

- **Error: Cannot open camera**: Verifica que el Webcam ID sea correcto
- **Ventana no responde**: Presiona 'q' para salir de forma segura
- **Rendimiento lento**: Reduce el inference size o cierra otras aplicaciones

## 📚 Más Información

Para documentación completa, consulta:
- `README.md`: Documentación general
- `GUI_IMPROVEMENTS.md`: Detalles de las mejoras de la GUI
