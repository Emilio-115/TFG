import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os

def _matrix_to_base64(matrix, title):
    """Genera la imagen de la matriz y la codifica en base64."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred False', 'Pred True'],
                yticklabels=['Actual False', 'Actual True'])
    plt.title(f'Matriz de Confusión: {title}')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_html_report(results, file_name="reporte_final.html", context=""):
    """
    Genera el reporte en results/base compatible con la clase Results y diccionarios.
    """
    if not file_name.lower().endswith(".html"):
        file_name += ".html"

    # Ruta absoluta para Windows (evita problemas con espacios en rutas de TFG)
    destiny_path = os.path.join(os.getcwd(), "results", "base")
    os.makedirs(destiny_path, exist_ok=True)
    full_path = os.path.join(destiny_path, file_name)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; margin: 30px; background-color: #f4f7f6; color: #333; }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
            .contexto {{ text-align: center; color: #666; font-style: italic; margin-bottom: 40px; font-size: 1.1em; }}
            .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h2 {{ color: #2980b9; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: center; }}
            .table-wrap {{ flex: 1; min-width: 450px; overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; white-space: nowrap; }}
            th {{ background-color: #2980b9; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Reporte de resultados de Enfermedades</h1>
        <div class="contexto">{context}</div>
    """

    for prolapse, result in results.items():
        # Lógica para extraer datos si es objeto Results o Diccionario
        if hasattr(result, 'conf_matrix'):
            matrix = result.conf_matrix
            report = result.classif_report
        else:
            # Fallback por si en algún momento vuelves a usar diccionarios
            matrix = result.get('Matrix') or result.get('conf_matrix')
            report = result.get('Report') or result.get('classif_report')

        img_str = _matrix_to_base64(matrix, prolapse)
        df_reporte = pd.DataFrame(report).transpose().round(5)
        title = prolapse.replace('_', ' ').upper()

        html += f"""
        <div class="card">
            <h2>{title}</h2>
            <div class="container">
                <div style="flex: 0 0 400px;">
                    <img src="data:image/png;base64,{img_str}" width="100%">
                </div>
                <div class="table-wrap">
                    {df_reporte.to_html()}
                </div>
            </div>
        </div>
        """

    html += "</body></html>"

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✅ Reporte guardado en: {full_path}")