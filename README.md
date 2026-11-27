# La ciencia en columnas de opinión colombianas (2018–2020)

Prototipo de visualización desarrollado como parte del trabajo de grado *Análisis de percepción de la ciencia en el discurso público colombiano con metodologías de minería de texto* (Maestría en Ciencia de los Datos y Analítica, Universidad EAFIT).

Permite explorar las menciones a la ciencia identificadas en 13.676 columnas de opinión de El Espectador, El Tiempo y Semana mediante búsqueda semántica con las categorías del Tesauro de la UNESCO.

## Pestañas
- **Panorama**: distribución por categoría, periódico y mes.
- **Tópicos**: tópicos del subcorpus de ciencia y su evolución mensual.
- **Entidades**: personas, organizaciones y lugares asociados a la ciencia.
- **Autores**: presencia de la ciencia por columnista.
- **Fragmentos**: búsqueda en los fragmentos clasificados.

## Datos
La carpeta `data/` contiene resultados precalculados del análisis. Por derechos de autor, los fragmentos se muestran recortados a 50 palabras.

## Ejecución local
pip install -r requirements.txt
streamlit run app.py