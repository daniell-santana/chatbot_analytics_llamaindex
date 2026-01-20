import pandas as pd
import asyncio
import numpy as np
# CONFIGURAÇÃO PARA MOSTRAR TODAS AS LINHAS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

from fastapi import FastAPI, Response
from datetime import datetime, date, timedelta 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data_loader import DataLoader
from sales_analyst import SalesDataAnalyst
import os
import uvicorn
from dotenv import load_dotenv
from api_cost_tracker import cost_tracker

#  1. Definição do Modelo de Requisição 
class ChatRequest(BaseModel):
    message: str
    filters: dict = {}

#  2. Configuração de Variáveis de Ambiente 
load_dotenv()
FILE_PATH = os.getenv('FILE_PATH', 'data/sales_dataset.csv')

#  3. Inicialização dos Componentes 
try:
    print(f"📊 Carregando dados de: {FILE_PATH}...")
    data_loader = DataLoader(FILE_PATH)
    df = data_loader.processed_data()
    analista = SalesDataAnalyst(df)
    print("✅ Dados e Analista carregados com sucesso!")
    
    # Diagnóstico inicial
    print(f"\n📈 Resumo dos dados:")
    print(f"   • Período: {df['ano'].min()} - {df['ano'].max()}")
    print(f"   • Total de vendas: R$ {df['valor_total'].sum():,.2f}")
    print(f"   • Registros: {len(df):,}")
    print(f"   • Categorias: {df['categoria'].nunique()}")
    print(f"   • Regiões: {df['regiao'].nunique()}")
    
except Exception as e:
    print(f"❌ ERRO ao carregar dados: {e}")
    analista = None
    
#  4. Aplicativo FastAPI 
app = FastAPI(
    title="SalesInsight AI API",
    description="API para análise de dados de vendas via LLM"
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Executar código pandas 
def executar_e_capturar_resultado(pandas_code_str, pergunta_original):
    """Executa o código pandas e captura o resultado completo."""
    try:
        import pandas as pd
        import numpy as np
        import warnings
        import ast
        
        # Suprime warnings específicos
        warnings.filterwarnings('ignore', message='Boolean Series key will be reindexed')
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Usa o dataframe global
        global df
        
        print(f"🔍 Código RAW recebido do LlamaIndex:\n{pandas_code_str[:500]}...")
        
        # 1. Extrai o código limpo (remove markdown)
        pandas_code = pandas_code_str
        
        if '```python' in pandas_code:
            code_start = pandas_code.find('```python') + 9
            code_end = pandas_code.find('```', code_start)
            pandas_code = pandas_code[code_start:code_end].strip()
            print(f"   🔄 Extraído código de bloco markdown")
        elif '```' in pandas_code:
            code_start = pandas_code.find('```') + 3
            code_end = pandas_code.find('```', code_start)
            pandas_code = pandas_code[code_start:code_end].strip()
            print(f"   🔄 Extraído código de bloco simples")
        
        # 2. Limpeza básica
        pandas_code = pandas_code.strip()
        
        if (pandas_code.startswith('"') and pandas_code.endswith('"')) or \
           (pandas_code.startswith("'") and pandas_code.endswith("'")):
            pandas_code = pandas_code[1:-1]
            print(f"   🔄 Aspas externas removidas")
        
        if pandas_code.endswith(';'):
            pandas_code = pandas_code[:-1]
            print(f"   🔄 Ponto-e-vírgula final removido")

        # REMOVE PRINT() SE EXISTIR
        if 'print(' in pandas_code and pandas_code.strip().startswith('print('):
            # Extrai o conteúdo dentro do print
            pandas_code = pandas_code.replace('print(', '').rstrip(')')
            print("   🔄 Removido print() do código")
        
        print(f"🔍 Código antes da correção de indentação:\n{pandas_code}")
        
        # 2.5. CORRIGE INDENTAÇÃO - Solução robusta para código pandas multi-linha
        def _corrigir_indentacao_pandas(code):
            """Corrige problemas de indentação em código pandas multi-linha."""
            
            # Se não tem quebra de linha, retorna como está
            if '\n' not in code:
                return code
            
            lines = code.split('\n')
            
            # Se tem poucas linhas, tenta juntar como uma linha
            if len(lines) <= 5:
                # Tenta juntar como linha única
                single_line = ' '.join([line.strip() for line in lines if line.strip()])
                
                # Remove espaços múltiplos
                import re
                single_line = re.sub(r'\s+', ' ', single_line)
                
                # Verifica se a linha única é válida
                try:
                    # Teste básico de sintaxe
                    ast.parse(single_line)
                    print(f"   🔄 Convertido para linha única ({len(lines)} linhas → 1 linha)")
                    return single_line
                except SyntaxError:
                    print(f"   ⚠️ Não conseguiu converter para linha única, mantendo multi-linha")
            
            # Para código multi-linha, corrige indentação
            cleaned_lines = []
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue  # Ignora linhas vazias
                
                # Se linha começa com '.' (método chain), junta com anterior
                if line_stripped.startswith('.'):
                    if cleaned_lines:
                        # Remove qualquer espaçamento antes do ponto
                        line_stripped = line_stripped.lstrip()
                        # Junta com linha anterior
                        cleaned_lines[-1] = cleaned_lines[-1] + ' ' + line_stripped
                        continue
                
                # Remove indentação de espaços
                if line.startswith('  ') or line.startswith(' ') or line.startswith('\t'):
                    # Conta espaços no início
                    space_count = len(line) - len(line.lstrip())
                    # Remove apenas a indentação comum (2 espaços)
                    if space_count >= 2:
                        line_stripped = line[2:].lstrip()
                    else:
                        line_stripped = line.lstrip()
                
                cleaned_lines.append(line_stripped)
            
            # Tenta como linha única primeiro
            single_line_attempt = ' '.join(cleaned_lines)
            try:
                ast.parse(single_line_attempt)
                print(f"   🔄 Código multi-linha convertido para linha única")
                return single_line_attempt
            except SyntaxError:
                # Se falhar, mantém como multi-linha corrigido
                corrected_code = '\n'.join(cleaned_lines)
                print(f"   ⚠️ Mantendo como código multi-linha corrigido")
                return corrected_code
        
        # Aplica correção de indentação
        pandas_code_corrigido = _corrigir_indentacao_pandas(pandas_code)
        
        # Se ainda tiver quebras de linha, força para linha única
        if '\n' in pandas_code_corrigido:
            print(f"   ⚠️ Ainda tem quebras de linha, forçando para linha única...")
            # Remove quebras e espaços extras
            pandas_code_corrigido = ' '.join(pandas_code_corrigido.split())
            print(f"   🔄 Forçado para linha única: {pandas_code_corrigido[:100]}...")
        
        pandas_code = pandas_code_corrigido
        
        print(f"🔍 Código após correção de indentação:\n{pandas_code}")
        print(f"🔍 Código limpo para execução (primeiros 200 chars):\n{pandas_code[:200]}...")
        
        # 3. Ambiente de execução
        namespace = {
            'df': df,
            'pd': pd,
            'np': np,
            '__builtins__': __builtins__,
            '__name__': '__main__'
        }
        
        resultado = None
        
        # 4. Tenta executar com eval primeiro (para capturar retorno direto)
        try:
            resultado = eval(pandas_code, namespace)
            print(f"✅ Resultado via eval: {type(resultado)}")
        except Exception as eval_err:
            print(f"⚠️ Eval falhou: {eval_err}, tentando exec...")
            try:
                # Executa o código
                exec(pandas_code, namespace)
                
                # Procura por variáveis de resultado comuns
                possible_result_names = ['__result', 'result', 'res', 'df_result', 'data']
                for var_name in possible_result_names:
                    if var_name in namespace:
                        resultado = namespace[var_name]
                        print(f"   🔍 Resultado encontrado como: {var_name}")
                        break
                
                # Se não encontrou, procura por qualquer variável que não seja padrão
                if resultado is None:
                    exclude_vars = ['df', 'pd', 'np', '__builtins__', '__name__', 
                                   '__warningregistry__', '__file__', '__doc__',
                                   '__package__', '__loader__', '__spec__', 'warnings']
                    
                    for var_name, var_value in namespace.items():
                        if (var_name not in exclude_vars and 
                            not var_name.startswith('_') and
                            not callable(var_value) and
                            not isinstance(var_value, type)):
                            
                            # Verifica se é um resultado válido (não string vazia, não None)
                            if var_value is not None and var_value != '':
                                resultado = var_value
                                print(f"   🔍 Resultado encontrado: {var_name}")
                                break
                                
            except Exception as exec_err:
                print(f"❌ Exec também falhou: {exec_err}")
                resultado = None
        
        # 5. Se ainda não tem resultado, tenta uma última abordagem
        if resultado is None:
            print(f"🔄 Última tentativa: exec com captura explícita...")
            try:
                # Executa com uma variável explícita para capturar
                exec_code = f"_temp_result = {pandas_code}"
                exec(exec_code, namespace)
                if '_temp_result' in namespace:
                    resultado = namespace['_temp_result']
                    print(f"✅ Resultado capturado via _temp_result")
            except Exception as last_err:
                print(f"❌ Última tentativa falhou: {last_err}")
        
        # 6. Log e retorno do resultado
        if resultado is not None:
            print(f"✅ Resultado final: {type(resultado)}")
            if isinstance(resultado, pd.DataFrame):
                print(f"   📋 Shape: {resultado.shape}")
            elif isinstance(resultado, pd.Series):
                print(f"   📊 Tamanho: {len(resultado)}")
            elif isinstance(resultado, dict):
                print(f"   📝 Dicionário com {len(resultado)} chaves")
        
        return resultado
        
    except Exception as e:
        print(f"⚠️ Erro ao executar código: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def executar_codigo_visualizacao(code_str, pergunta):
    """Executa código de visualização (matplotlib, seaborn, plotly, etc.) e captura a figura."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Sem isso, o servidor FastAPI VAI FALHAR quando tentar gerar gráficos
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        import io
        import base64
        import pandas as pd
        import numpy as np
        
        # Usa o dataframe global
        global df
        
        print(f"🎨 Executando código de visualização...")
        print(f"🔍 Tipo de código detectado: {'Matplotlib' if 'plt.' in code_str.lower() else 'Seaborn' if 'sns.' in code_str.lower() else 'Plotly' if 'plotly' in code_str.lower() else 'Pandas plotting'}")
        
        # Extrai o código limpo (remove markdown)
        clean_code = code_str
        if '```python' in clean_code:
            code_start = clean_code.find('```python') + 9
            code_end = clean_code.find('```', code_start)
            clean_code = clean_code[code_start:code_end].strip()
        elif '```' in clean_code:
            code_start = clean_code.find('```') + 3
            code_end = clean_code.find('```', code_start)
            clean_code = clean_code[code_start:code_end].strip()
        
        # Remove comandos que não funcionam em backend não-interativo
        clean_code = clean_code.replace('plt.show()', '')
        clean_code = clean_code.replace('plt.show(', '# plt.show(')
        clean_code = clean_code.replace('fig.show()', '# fig.show()')
        clean_code = clean_code.replace('.show()', '# .show()')
        
        # Cria namespace com todas as bibliotecas de visualização
        namespace = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'range': range,
            'len': len,
            'print': print,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'bool': bool,
            'type': type,
            'isinstance': isinstance,
            'round': round,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs
        }
        
        # Executa o código
        exec(clean_code, namespace)
        
        # TENTA DIFERENTES FORMAS DE CAPTURAR A FIGURA
        
        # 1. Se for matplotlib/seaborn
        if 'plt.' in clean_code.lower() or 'sns.' in clean_code.lower():
            # Salva a figura atual do matplotlib
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            plt.close('all')  # Fecha todas as figuras
            buf.seek(0)
            
            # Converte para base64
            img_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # 2. Se for plotly
        elif 'plotly' in clean_code.lower() or 'px.' in clean_code.lower() or 'go.' in clean_code.lower():
            # Procura a figura plotly no namespace
            fig = None
            for var_name, var_value in namespace.items():
                if hasattr(var_value, 'to_image'):
                    fig = var_value
                    break
            
            if fig:
                # Converte plotly para imagem
                buf = io.BytesIO()
                fig.write_image(buf, format='png', width=800, height=500)
                buf.seek(0)
                img_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            else:
                raise ValueError("Não foi possível encontrar figura plotly no código")
        
        # Se for .plot.pie
        elif '.plot.pie(' in clean_code.lower() or '.pie(' in clean_code.lower():
            # Para pandas plotting de pizza
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            plt.close('all')
            buf.seek(0)
            img_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # 3. Se for pandas plotting
        elif '.plot(' in clean_code.lower():
            # Para pandas plotting, plt.gcf() deve ter a figura
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            plt.close('all')
            buf.seek(0)
            img_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        else:
            raise ValueError("Tipo de visualização não suportado")
        
        print(f"✅ Visualização gerada com sucesso!")
        return img_base64
        
    except Exception as e:
        print(f"❌ Erro ao executar código de visualização: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: tenta executar apenas a parte pandas e criar gráfico básico
        try:
            print(f"🔄 Tentando fallback: extraindo dados e criando gráfico básico...")
            
            # Extrai apenas a parte de agregação de dados
            lines = clean_code.split('\n')
            data_lines = []
            for line in lines:
                if 'df[' in line and ('.groupby(' in line or '.query(' in line or 'df[' in line):
                    data_lines.append(line)
                elif not any(keyword in line for keyword in ['plt.', 'sns.', 'px.', '.plot(', '.show()']):
                    data_lines.append(line)
            
            data_code = '\n'.join(data_lines)
            
            # Executa apenas para obter dados
            namespace_simple = {'df': df, 'pd': pd, 'np': np}
            exec(data_code, namespace_simple)
            
            # Procura o resultado
            result = None
            for var_name, var_value in namespace_simple.items():
                if var_name not in ['df', 'pd', 'np'] and var_name != '__builtins__':
                    if isinstance(var_value, (pd.DataFrame, pd.Series)):
                        result = var_value
                        break
            
            if result is not None:
                import matplotlib.pyplot as plt_fallback
                import io
                import base64
                
                # Cria gráfico básico
                plt_fallback.figure(figsize=(10, 6))
                
                if isinstance(result, pd.Series):
                    result.plot(kind='bar' if len(result) < 15 else 'line')
                    plt_fallback.title(f"Resultado: {result.name if hasattr(result, 'name') else 'Série'}")
                elif isinstance(result, pd.DataFrame):
                    if len(result.columns) == 1:
                        result.iloc[:, 0].plot(kind='bar' if len(result) < 15 else 'line')
                    else:
                        result.plot(kind='line')
                    plt_fallback.title(f"Resultado: {len(result)} registros")
                
                plt_fallback.tight_layout()
                
                # Salva
                buf = io.BytesIO()
                plt_fallback.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt_fallback.close('all')
                buf.seek(0)
                
                img_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                print(f"✅ Fallback: gráfico básico gerado a partir dos dados")
                return img_base64
        
        except Exception as fallback_error:
            print(f"❌ Fallback também falhou: {fallback_error}")
        
        return None
    
def formatar_resultado_pandas(resultado, pergunta_original):
    """Formata qualquer resultado do Pandas para exibição HTML limpa e responsiva."""
    try:
        import pandas as pd
        import numpy as np
        
        # DEBUG DETALHADO
        print(f"🔍 FORMATANDO RESULTADO - DEBUG:")
        print(f"   Tipo: {type(resultado)}")
        print(f"   É None? {resultado is None}")
        
        if resultado is not None:
            if hasattr(resultado, 'shape'):
                print(f"   Shape: {resultado.shape}")
            if hasattr(resultado, '__len__'):
                print(f"   Length: {len(resultado)}")
            print(f"   Valor exemplo: {str(resultado)[:100]}...")
        
        print(f"   Pergunta: {pergunta_original}")
        
        # 1. VALIDAÇÕES INICIAIS
        if resultado is None:
            print("⚠️ Resultado é None, retornando string vazia")
            return ""  # Retorna string vazia, não mensagem
        
        # Se o resultado já é uma string HTML, retorna diretamente
        if isinstance(resultado, str) and ('<table' in resultado or '<div' in resultado):
            print("✅ Resultado já é HTML, retornando diretamente")
            return resultado
        
        # 2.5. VERIFICAÇÃO ESPECIAL PARA DATAFRAMES/Series
        # O resultado pode vir como string do pandas, precisamos converter
        if isinstance(resultado, str):
            # Tenta detectar se é uma representação de DataFrame/Series
            if "DataFrame" in resultado or "Series" in resultado or "trimestre" in resultado.lower():
                print(f"⚠️ Resultado é string mas parece DataFrame/Series: {resultado[:100]}...")
                # Tenta extrair os dados
                try:
                    # Executa o código novamente para obter o objeto
                    global df
                    namespace = {'df': df, 'pd': pd, 'np': np}
                    # Procura por código pandas na string
                    import re
                    code_match = re.search(r'df\[.*\]', resultado)
                    if code_match:
                        code = code_match.group()
                        exec(f"__temp_result = {code}", namespace)
                        resultado = namespace.get('__temp_result')
                        print(f"🔧 Convertido string para objeto: {type(resultado)}")
                except Exception as conv_error:
                    print(f"❌ Não foi possível converter string: {conv_error}")
                    # Se não conseguiu converter, mantém como string
                    pass
        
        print(f"✅ Tipo final para formatação: {type(resultado)}")
        
        # FUNÇÕES AUXILIARES INTERNAS
        def _formatar_valor(valor, col_name=""):
            """Formata qualquer valor de forma inteligente, com contexto do nome da coluna."""
            if pd.isna(valor):
                return "-"
            
            # Primeiro, verifica se é número
            if isinstance(valor, (int, float, np.number)):
                # DETECÇÃO DE PORCENTAGEM (crescimento, percentual, taxa, índice)
                col_lower = col_name.lower() if col_name else ""
                
                # Verifica se a coluna indica porcentagem
                is_percent_column = any(term in col_lower for term in 
                                    ['percentual', 'crescimento', 'taxa', 'índice', 'margem', '%'])
                
                if is_percent_column:
                    # Se o número for entre 0 e 1, converte para porcentagem (ex: 0.1234 → 12.34%)
                    if 0 <= abs(valor) <= 1:
                        return f"{valor*100:.2f}%"
                    # Se já estiver como porcentagem (ex: 12.34), apenas adiciona %
                    else:
                        return f"{valor:.2f}%"
                
                # Moeda - para valores grandes
                is_count_column = any(keyword in col_lower for keyword in 
                                    ['numero', 'número', 'quantidade', 'qtd', 'contagem', 'count'])
                
                if abs(valor) >= 100 and not is_count_column:
                    if isinstance(valor, float) or (isinstance(valor, (int, np.integer)) and valor >= 1000):
                        valor_formatado = f"R$ {valor:,.2f}"
                        return valor_formatado.replace(",", "X").replace(".", ",").replace("X", ".")
                
                # Números inteiros grandes
                if isinstance(valor, (int, np.integer)) and abs(valor) > 999:
                    return f"{valor:,}".replace(",", ".")
                
                # Números decimais
                if isinstance(valor, float):
                    return f"{valor:.2f}"
                
                return str(valor)
            
            # Strings
            elif isinstance(valor, str):
                return valor
            
            # Outros tipos
            return str(valor)
        
        def _formatar_mes(numero):
            """Converte número do mês para nome, se aplicável."""
            try:
                # Se for string que já começa com 'T', mantém como está (TRIMESTRE)
                if isinstance(numero, str) and numero.startswith('T'):
                    return numero
                    
                # Se for número, converte
                if isinstance(numero, (int, float, np.integer, np.floating)):
                    num = int(float(numero))
                    
                    # VERIFICA SE É TRIMESTRE (1-4) - mantém como T1, T2, T3, T4
                    # Só converte para nome do mês se estiver entre 1-12 E não for especificamente trimestre
                    meses = {
                        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 
                        5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
                        9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
                    }
                    
                    # Se for número 1-12, pode ser mês OU trimestre
                    # Depende do contexto - por padrão, assume que é mês
                    if 1 <= num <= 12:
                        return meses.get(num, str(numero))
                    else:
                        return str(numero)
                
                # Se for string que pode ser convertida para número
                elif isinstance(numero, str) and numero.isdigit():
                    num = int(numero)
                    if 1 <= num <= 12:
                        meses = {
                            1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 
                            5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
                            9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
                        }
                        return meses.get(num, numero)
                
                return str(numero)
                    
            except:
                return str(numero)
                
        def _dataframe_fallback_simples(df_obj):
            """Fallback simples para DataFrames quando a formatação principal falha."""
            try:
                # Tenta converter para HTML simples
                html = df_obj.to_html(
                    index=False if hasattr(df_obj, 'reset_index') else True,
                    classes='dataframe-table',
                    border=0,
                    float_format=lambda x: f'{x:,.2f}' if isinstance(x, (int, float)) else str(x)
                )
                
                # Adiciona estilos básicos
                html = f'''
                <div style="overflow-x: auto; margin: 10px 0;">
                    <style>
                        .dataframe-table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                            font-family: Arial, sans-serif;
                            font-size: 12px;
                        }}
                        .dataframe-table th {{
                            background: #f8f9fa;
                            padding: 8px 12px;
                            text-align: left;
                            border-bottom: 2px solid #ddd;
                        }}
                        .dataframe-table td {{
                            padding: 6px 12px;
                            border-bottom: 1px solid #eee;
                        }}
                        .dataframe-table tr:hover {{ background: #f5f5f5; }}
                    </style>
                    {html}
                </div>
                '''
                return html
            except Exception as e:
                print(f"❌ Fallback também falhou: {e}")
                return str(df_obj)
        
        # 3. TRATAMENTO POR TIPO DE RESULTADO
        # ========== DATA FRAME ==========
        if isinstance(resultado, pd.DataFrame):
            print(f"📊 Formatando DataFrame: {len(resultado)}x{len(resultado.columns)}")
            
            if resultado.empty:
                return ""
            
            try:
                # Copia para não modificar original
                df_display = resultado.copy()
                
                # ========== PRIMEIRO: DECISÃO DE RESETAR ÍNDICE ==========
                print(f"   🔍 Tipo de índice: {type(df_display.index)}")
                print(f"   🔍 Nome do índice: {df_display.index.name}")
                print(f"   🔍 Valores do índice: {df_display.index.tolist()[:10] if len(df_display) > 10 else df_display.index.tolist()}")
                
                # DECISÃO: Quando resetar o índice?
                if df_display.index.name is not None:
                    # Índice tem nome (ex: 'regiao', 'categoria') → reseta para incluir na tabela
                    print(f"   🔍 Índice tem nome: '{df_display.index.name}' → resetando")
                    df_display = df_display.reset_index()
                elif isinstance(df_display.index, pd.MultiIndex):
                    print(f"   🔍 Índice é MultiIndex → resetando")
                    df_display = df_display.reset_index()
                elif not isinstance(df_display.index, pd.RangeIndex):
                    print(f"   🔍 Índice não é RangeIndex → resetando")
                    df_display = df_display.reset_index()
                elif len(df_display) > 0:
                    # Verifica se é um RangeIndex padrão (0, 1, 2, ...)
                    expected_index = pd.RangeIndex(start=0, stop=len(df_display))
                    if not df_display.index.equals(expected_index):
                        print(f"   🔍 RangeIndex não padrão → resetando")
                        df_display = df_display.reset_index()
                    else:
                        print(f"   🔍 RangeIndex padrão (0, 1, 2, ...) → mantendo")
                        # NÃO reseta - mantém o índice oculto
                else:
                    print(f"   🔍 DataFrame vazio ou pequeno → mantendo como está")
                
                print(f"   ✅ Colunas após decisão: {df_display.columns.tolist()}")
                
                # ========== SEGUNDO: REMOVER COLUNA 'index' SE FOR SEQUENCIAL ==========
                if 'index' in df_display.columns:
                    # Verifica se é uma sequência numérica simples
                    is_sequential_index = all(
                        isinstance(df_display.at[i, 'index'], (int, np.integer)) and 
                        df_display.at[i, 'index'] == i 
                        for i in range(len(df_display))
                    )
                    
                    if is_sequential_index:
                        print(f"   🔍 Removendo coluna 'index' (índice sequencial)")
                        df_display = df_display.drop(columns=['index'])
                    else:
                        # Renomeia para algo mais descritivo
                        df_display = df_display.rename(columns={'index': 'Posição'})
                
                print(f"   ✅ Colunas após limpeza: {df_display.columns.tolist()}")
                
                # ========== TERCEIRO: TRATAMENTO ESPECIAL PARA TABELAS PIVOTADAS ==========
                # Verifica se é uma tabela com anos como colunas (ex: 2020, 2021, 2022...)
                are_years_columns = all(
                    isinstance(col, (int, float)) and 2000 <= col <= 2100 
                    for col in df_display.columns 
                    if col != 'regiao' and col != '...' and str(col).isdigit()
                )
                
                if are_years_columns and 'regiao' in df_display.columns:
                    print(f"   🔍 Detectada tabela pivotada com anos como colunas")
                    # Reordena colunas: regiao primeiro, depois anos
                    year_cols = [col for col in df_display.columns if col != 'regiao' and col != '...']
                    df_display = df_display[['regiao'] + year_cols]
                    
                    # Renomeia anos para não ter "T" na frente (anos não são trimestres!)
                    rename_years = {}
                    for col in df_display.columns:
                        if isinstance(col, (int, float)) and 2000 <= col <= 2100:
                            rename_years[col] = str(int(col))
                    df_display = df_display.rename(columns=rename_years)
                
                # Limita tamanho para exibição
                max_rows_display = 15
                max_cols_display = 8
                
                if len(df_display) > max_rows_display:
                    df_display = pd.concat([
                        df_display.head(max_rows_display // 2),
                        pd.DataFrame([["..."] * len(df_display.columns)], columns=df_display.columns),
                        df_display.tail(max_rows_display // 2)
                    ], ignore_index=True)
                    print(f"   ⚠️ DataFrame truncado para {max_rows_display} linhas")
                
                if len(df_display.columns) > max_cols_display:
                    df_display = df_display.iloc[:, :max_cols_display]
                    df_display["..."] = "..."
                    print(f"   ⚠️ DataFrame truncado para {max_cols_display} colunas")
                
                # ========== QUARTO: Renomeia colunas para melhor legibilidade ==========
                rename_map = {}
                for col in df_display.columns:
                    # Converte col para string para evitar erros
                    col_str = str(col)
                    
                    if col_str == "...":
                        rename_map[col] = col_str
                    elif col_str.isdigit() and len(col_str) == 4 and 2000 <= int(col_str) <= 2100:
                        # É um ano (ex: 2023) → mantém como número, não converte para T2023
                        rename_map[col] = col_str
                    elif col_str.isdigit():  # Se for número (como trimestres 1, 2, 3, 4)
                        rename_map[col] = f"T{col_str}"  # Converte para T1, T2, etc.
                    elif col_str.lower() in ['regiao', 'região', 'region']:
                        rename_map[col] = 'Região'
                    elif col_str.lower() in ['trimestre', 'quarter']:
                        rename_map[col] = 'Trimestre'
                    elif col_str.lower() in ['mes', 'mês', 'month']:
                        rename_map[col] = 'Mês'
                    elif col_str.lower() in ['ano', 'year']:
                        rename_map[col] = 'Ano'
                    elif col_str.lower() in ['categoria', 'category']:
                        rename_map[col] = 'Categoria'
                    elif col_str.lower() in ['produto', 'product']:
                        rename_map[col] = 'Produto'
                    elif col_str.lower() in ['valor', 'valor_total', 'total']:
                        rename_map[col] = 'Valor'
                    elif col_str.lower() in ['quantidade', 'qtd']:
                        rename_map[col] = 'Qtd'
                    elif col_str.lower() in ['lucro', 'profit']:
                        rename_map[col] = 'Lucro'
                    elif col_str.lower() in ['margem', 'margem_lucro']:
                        rename_map[col] = 'Margem'
                    elif 'forma_pagamento' in col_str.lower() or 'pagamento' in col_str.lower():
                        rename_map[col] = 'Forma de Pagamento'
                    elif 'ticket' in col_str.lower():
                        rename_map[col] = 'Ticket Médio'
                    elif 'numero' in col_str.lower() or 'número' in col_str.lower():
                        rename_map[col] = 'Número de Vendas'
                    else:
                        rename_map[col] = str(col).title()
                
                df_display = df_display.rename(columns=rename_map)
                
                # ========== QUINTO: Formata os valores nas células ==========
                for col in df_display.columns:
                    # Pula colunas de placeholder
                    if col == "...":
                        continue
                    
                    # Converte nome da coluna para string
                    col_name = str(col).lower()
                    
                    for i in range(len(df_display)):
                        val = df_display.at[i, col]
                        
                        # Placeholder de truncamento
                        if isinstance(val, str) and val == "...":
                            continue
                        
                        # DECISÃO DE FORMATAÇÃO BASEADA NO NOME DA COLUNA
                        formatted_val = None
                        
                        # 1. ANOS - prioridade máxima
                        if ('ano' in col_name or 'year' in col_name) and isinstance(val, (int, np.integer)):
                            if 1900 <= val <= 2100:  # Faixa razoável para anos
                                formatted_val = str(int(val))

                        # 2. TRIMESTRES - prioridade alta
                        # Detecção: Verifica se a coluna é 'trimestre' ou se o valor é 1-4
                        if formatted_val is None and ('trimestre' in col_name.lower() or 'quarter' in col_name.lower()) and isinstance(val, (int, float, np.integer, np.floating)):
                            # Formata trimestres como T1, T2, T3, T4
                            try:
                                if 1 <= float(val) <= 4:
                                    formatted_val = f"T{int(float(val))}"
                            except:
                                pass
                        
                        # 2.5. DETECÇÃO ESPECIAL: Se é uma tabela de trimestres (nomes de colunas incluem anos)
                        # Neste caso, o valor na coluna 'Trimestre' deve ser T1, T2, etc.
                        if formatted_val is None and col_name.lower() == 'trimestre' and isinstance(val, (int, float, np.integer, np.floating)):
                            # Verifica se estamos em uma tabela de comparação (tem anos como colunas)
                            is_comparison_table = any(
                                isinstance(str(c), str) and str(c).isdigit() and len(str(c)) == 4 and 2000 <= int(str(c)) <= 2100
                                for c in df_display.columns
                            )
                            
                            if is_comparison_table:
                                # Em tabela de comparação, trimestres devem ser T1, T2, etc.
                                try:
                                    if 1 <= float(val) <= 4:
                                        formatted_val = f"T{int(float(val))}"
                                except:
                                    pass
                                                                
                        # 3. MESES - prioridade média
                        if formatted_val is None and ('mes' in col_name.lower() or 'mês' in col_name.lower()) and isinstance(val, (int, float, np.integer, np.floating)):
                            month_name = _formatar_mes(val)
                            if month_name != str(val):  # Se conseguiu converter
                                formatted_val = month_name
                        
                        # 4. CONTAGENS/NÚMEROS INTEIROS
                        if formatted_val is None and any(keyword in col_name for keyword in ['numero', 'número', 'quantidade', 'qtd', 'contagem', 'count', 'vendas', 'regsitro']):
                            if isinstance(val, (int, float, np.number)):
                                # É número inteiro (contagem)
                                if isinstance(val, (int, np.integer)) or val == int(val):
                                    formatted_val = f"{int(val):,}".replace(",", ".")
                                else:
                                    formatted_val = f"{val:.0f}"
                        
                        # 5. PORCENTAGENS
                        if formatted_val is None and any(keyword in col_name for keyword in ['margem', 'porcentagem', 'percentual', '%', 'taxa', 'índice', 'indice']):
                            if isinstance(val, (int, float, np.number)):
                                valor_float = float(val)
                                # Se o valor está entre 0 e 1, converte para porcentagem
                                if 0 <= valor_float <= 1:
                                    formatted_val = f"{valor_float*100:.1f}%"
                                # Se já está em porcentagem (ex: 15.5)
                                elif 0 <= valor_float <= 100:
                                    formatted_val = f"{valor_float:.1f}%"
                                # Se for um valor monetário grande, NÃO é porcentagem
                                # (deixa para formatação genérica abaixo)
                        
                        # 6. Se não encontrou formatação especial, usa a genérica COM contexto
                        if formatted_val is None:
                            formatted_val = _formatar_valor(val, col_name=col_name)
                        
                        df_display.at[i, col] = formatted_val
                
                # ========== SEXTO: Gera HTML com estilos ==========
                html_table = df_display.to_html(
                    index=False,
                    classes='dataframe-table',
                    border=0,
                    na_rep='-',
                    escape=False
                )

                # DEBUG: Ver o que está sendo gerado
                print(f"   🔍 HTML gerado (primeiros 200 chars): {html_table[:200]}...")

                # VERSÃO SIMPLIFICADA: evita problemas com split()
                html_output = f'''
                <div style="
                    overflow-x: auto; 
                    margin: 10px 0;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                ">
                    {html_table}
                </div>
                '''
                
                # Aplica estilos inline (manualmente)
                html_output = html_output.replace(
                    '<thead>',
                    '''<thead>
                    <tr style="
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-bottom: 2px solid #dadce0;
                    ">'''
                )
                
                html_output = html_output.replace(
                    '<th>',
                    '<th style="padding: 12px 16px; text-align: left; font-weight: 600; color: #202124; border-right: 1px solid #e0e0e0;">'
                )
                
                html_output = html_output.replace(
                    '<td>',
                    '<td style="padding: 10px 16px; border-bottom: 1px solid #f5f5f5; border-right: 1px solid #f0f0f0;">'
                )
                
                # Adiciona hover effect
                if '</table>' in html_output:
                    html_output = html_output.replace(
                        '</table>',
                        '''</table>
                        <style>
                        .dataframe-table tr:hover td {
                            background-color: #f8f9fa;
                        }
                        </style>'''
                    )
                
                return html_output
                
            except Exception as df_error:
                print(f"⚠️ Erro na formatação DataFrame: {df_error}")
                import traceback
                traceback.print_exc()
                # Usa fallback
                return _dataframe_fallback_simples(resultado)
        
        # ========== SERIES ==========
        elif isinstance(resultado, pd.Series):
            print(f"📈 Formatando Series: {len(resultado)} valores")
            
            if resultado.empty:
                return ""
            
            try:
                # Limita para exibição
                max_items = 20
                items_to_display = resultado.head(max_items) if len(resultado) > max_items else resultado
                
                # DETECÇÃO MELHORADA: Verifica se o índice são meses
                # Verifica pelos valores do índice (1-12) E pelo nome da Series
                series_name_lower = str(resultado.name).lower() if hasattr(resultado, 'name') else ""
                is_month_series = False
                
                # Verifica se o nome indica meses
                if 'mes' in series_name_lower or 'mês' in series_name_lower:
                    is_month_series = True
                # Ou verifica pelos valores do índice
                elif all(isinstance(idx, (int, float)) and 1 <= idx <= 12 for idx in resultado.index):
                    is_month_series = True
                elif all(isinstance(idx, str) and idx.isdigit() and 1 <= int(idx) <= 12 for idx in resultado.index):
                    is_month_series = True
                
                html_lines = ['<ul style="margin: 8px 0; padding-left: 20px; list-style: none;">']
                
                for idx, val in items_to_display.items():
                    # Formata índice
                    idx_str = str(idx)
                    if is_month_series:
                        idx_str = _formatar_mes(idx)
                    
                    # Formata valor (com formatação monetária)
                    val_str = _formatar_valor(val)
                    
                    html_lines.append(f'''
                    <li style="
                        margin: 4px 0;
                        padding: 6px 10px;
                        background: #f8f9fa;
                        border-radius: 6px;
                        border-left: 3px solid #4285f4;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span style="font-weight: 500; color: #5f6368;">{idx_str}</span>
                        <span style="font-weight: 600; color: #202124;">{val_str}</span>
                    </li>
                    ''')
                
                if len(resultado) > max_items:
                    html_lines.append(f'''
                    <li style="
                        margin: 4px 0;
                        padding: 6px 10px;
                        background: #f1f3f4;
                        border-radius: 6px;
                        color: #5f6368;
                        font-style: italic;
                        font-size: 12px;
                    ">
                        + {len(resultado) - max_items} itens adicionais...
                    </li>
                    ''')
                
                html_lines.append('</ul>')
                
                return '\n'.join(html_lines)
                
            except Exception as series_error:
                print(f"⚠️ Erro na formatação Series: {series_error}")
                # Fallback para Series
                try:
                    # Converte para DataFrame simples
                    df_temp = resultado.reset_index()
                    df_temp.columns = ['Item', 'Valor']
                    return _dataframe_fallback_simples(df_temp)
                except:
                    return f'<div style="padding: 10px; background: #f8f9fa; border-radius: 6px;">{str(resultado)}</div>'
        
        # ========== DICIONÁRIO ==========
        elif isinstance(resultado, dict):
            print(f"📝 Formatando dicionário: {len(resultado)} chaves")
            
            if not resultado:
                return ""
            
            html_lines = ['<div style="margin: 10px 0;">']
            
            for key, value in resultado.items():
                key_str = str(key)
                val_str = _formatar_valor(value)
                
                html_lines.append(f'''
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 6px 0;
                    padding: 8px 12px;
                    background: #f8f9fa;
                    border-radius: 6px;
                    border: 1px solid #e0e0e0;
                ">
                    <div style="font-weight: 500; color: #5f6368; min-width: 150px;">{key_str}</div>
                    <div style="font-weight: 600; color: #202124; text-align: right;">{val_str}</div>
                </div>
                ''')
            
            html_lines.append('</div>')
            
            return '\n'.join(html_lines)
        
        # ========== VALOR ESCALAR ==========
        elif isinstance(resultado, (int, float, np.number, str)):
            print(f"🔢 Formatando valor escalar: {resultado}")
            
            val_str = _formatar_valor(resultado)
            
            # Detecção inteligente do tipo de valor
            is_money = any(keyword in pergunta_original.lower() for keyword in 
                          ['valor', 'total', 'venda', 'lucro', 'custo', 'ticket', 'preço', 'montante'])
            is_percent = any(keyword in pergunta_original.lower() for keyword in 
                           ['porcentagem', 'percentual', '%', 'margem', 'taxa', 'crescimento'])
            
            if is_money and isinstance(resultado, (int, float, np.number)):
                return f'''
                <div style="
                    display: inline-block;
                    padding: 12px 20px;
                    background: linear-gradient(135deg, #34a853 0%, #0d652d 100%);
                    color: white;
                    border-radius: 10px;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 4px 12px rgba(52, 168, 83, 0.2);
                    text-align: center;
                    min-width: 120px;
                ">
                    R$ {float(resultado):,.2f}
                </div>
                '''.replace(',', 'X').replace('.', ',').replace('X', '.')
            
            elif is_percent and isinstance(resultado, (int, float, np.number)):
                percent_val = float(resultado) * 100 if float(resultado) <= 1 else float(resultado)
                return f'''
                <div style="
                    display: inline-block;
                    padding: 12px 20px;
                    background: linear-gradient(135deg, #4285f4 0%, #1a5fd6 100%);
                    color: white;
                    border-radius: 10px;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 4px 12px rgba(66, 133, 244, 0.2);
                    text-align: center;
                    min-width: 120px;
                ">
                    {percent_val:.1f}%
                </div>
                '''
            
            else:
                return f'''
                <div style="
                    display: inline-block;
                    padding: 12px 20px;
                    background: #f8f9fa;
                    color: #202124;
                    border-radius: 10px;
                    font-weight: 600;
                    font-size: 16px;
                    border: 1px solid #e0e0e0;
                    text-align: center;
                    min-width: 120px;
                ">
                    {val_str}
                </div>
                '''
        
        # ========== OUTROS TIPOS ==========
        else:
            print(f"⚡ Formatando tipo genérico: {type(resultado)}")
            
            result_str = str(resultado)
            
            # Se for muito longo, truncar
            if len(result_str) > 500:
                result_str = result_str[:497] + "..."
            
            # Limpa quebras de linha múltiplas
            result_str = ' '.join(result_str.splitlines())
            
            return f'''
            <div style="
                padding: 12px 16px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                font-family: monospace;
                font-size: 13px;
                color: #202124;
                word-break: break-word;
                max-height: 200px;
                overflow-y: auto;
            ">
                {result_str}
            </div>
            '''
    
    except Exception as e:
        print(f"⚠️ Erro na formatação: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback minimalista
        try:
            if isinstance(resultado, pd.DataFrame):
                return _dataframe_fallback_simples(resultado)
            elif isinstance(resultado, pd.Series):
                df_temp = resultado.reset_index()
                df_temp.columns = ['Item', 'Valor']
                return _dataframe_fallback_simples(df_temp)
            else:
                return f'<div style="padding: 10px; background: #f8f9fa; border-radius: 6px;">{str(resultado)[:300]}</div>'
        except:
            return f'<div style="padding: 10px; background: #f8f9fa; border-radius: 6px;">Resultado disponível</div>'
    
#  5. Endpoints da API 
@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "SalesInsight AI API rodando. Use POST /chat para consultas.",
        "endpoints": {
            "/": "Status da API",
            "/chat": "Consultas de análise (POST)",
            "/stats": "Estatísticas básicas (GET)"
        }
    }

@app.get("/stats")
def get_stats():
    """Retorna estatísticas básicas do dataset."""
    if analista is None:
        return {"error": "Analista não inicializado"}
    
    stats = {
        "total_vendas": f"R$ {df['valor_total'].sum():,.2f}",
        "total_registros": len(df),
        "periodo": f"{df['ano'].min()} - {df['ano'].max()}",
        "categorias": int(df['categoria'].nunique()),
        "produtos": int(df['produto'].nunique()),
        "regioes": int(df['regiao'].nunique()),
        "vendedores": int(df['vendedor'].nunique()),
        "ticket_medio": f"R$ {df['valor_total'].mean():,.2f}",
        "vendas_concluidas": len(df[df['status'] == 'CONCLUÍDA']),
        "taxa_cancelamento": f"{(len(df[df['status'] == 'CANCELADA']) / len(df) * 100):.2f}%"
    }
    return stats
# ============ ENDPOINT CUSTOS ============
@app.get("/api/costs")
async def get_api_costs(
    period: str = "today",
    format: str = "json"
):
    """
    Endpoint para obter dados de custos da API.
    
    Parâmetros:
    - period: today, week, month, all
    - format: json, csv
    """
    try:
        if period == "today":
            data = cost_tracker.get_daily_summary()
            
        elif period == "week":
            # Últimos 7 dias
            data = {
                "period": "last_7_days",
                "days": [],
                "total_cost": 0.0,
                "total_tokens": {"input": 0, "output": 0}
            }
            
            today = date.today()
            for i in range(7):
                day = today - timedelta(days=i)
                day_str = day.isoformat()
                daily = cost_tracker.get_daily_summary(day_str)
                
                if daily['cost'] > 0:
                    data["days"].append(daily)
                    data["total_cost"] += daily['cost']
                    data["total_tokens"]["input"] += daily['tokens']['input']
                    data["total_tokens"]["output"] += daily['tokens']['output']
            
        elif period == "month":
            today = datetime.now()
            data = cost_tracker.get_monthly_summary(today.year, today.month)
            
        elif period == "all":
            data = {
                "period": "all",
                "total_cost": cost_tracker.daily_stats.get("total_cost", 0.0),
                "total_tokens": cost_tracker.daily_stats.get("total_tokens", {"input": 0, "output": 0}),
                "model_usage": cost_tracker.daily_stats.get("model_usage", {}),
                "first_record": min(cost_tracker.daily_stats.get("daily_costs", {}).keys(), default=None),
                "last_record": max(cost_tracker.daily_stats.get("daily_costs", {}).keys(), default=None)
            }
        
        else:
            return {"error": "Periodo inválido. Use: today, week, month, all"}
        
        # Formata resposta
        if format == "csv":
            # Converte para CSV
            import pandas as pd
            if isinstance(data, dict) and "days" in data:
                df = pd.DataFrame(data["days"])
            else:
                df = pd.DataFrame([data])
            
            csv_data = df.to_csv(index=False)
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=costs.csv"}
            )
        
        else:  # JSON padrão
            return {
                "success": True,
                "period": period,
                "data": data,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {"error": str(e)}

# ========== FUNÇÕES AUXILIARES FLEXÍVEIS ==========

def _sugerir_eixo_x(chart_type, df):
    """Sugere eixo X baseado no tipo de gráfico e dados disponíveis."""
    # Prioridades por tipo de gráfico
    if chart_type in ["line", "bar"]:
        # Para séries temporais
        if 'mes' in df.columns and df['mes'].nunique() > 1:
            return 'mes'
        elif 'trimestre' in df.columns:
            return 'trimestre'
        elif 'ano' in df.columns:
            return 'ano'
        # Categóricos
        elif 'categoria' in df.columns:
            return 'categoria'
        elif 'regiao' in df.columns:
            return 'regiao'
    
    elif chart_type == "pie":
        if 'categoria' in df.columns:
            return 'categoria'
        elif 'regiao' in df.columns:
            return 'regiao'
    
    elif chart_type == "scatter":
        if 'valor_total' in df.columns:
            return 'valor_total'
    
    # Fallback: primeira coluna categórica
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    return categorical_cols[0] if categorical_cols else df.columns[0]

def _sugerir_eixo_y(chart_type, df):
    """Sugere eixo Y baseado no tipo de gráfico."""
    if 'valor_total' in df.columns:
        return 'valor_total'
    elif 'quantidade' in df.columns:
        return 'quantidade'
    elif 'lucro' in df.columns:
        return 'lucro'
    else:
        # Primeira coluna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

def _gerar_titulo_automatico(chart_type, x_axis, y_axis, filters):
    """Gera título automático baseado nos parâmetros."""
    chart_names = {
        "bar": "Gráfico de Barras",
        "line": "Gráfico de Linha",
        "pie": "Gráfico de Pizza",
        "scatter": "Gráfico de Dispersão"
    }
    
    base_title = f"{chart_names.get(chart_type, 'Gráfico')}: {y_axis} por {x_axis}"
    
    # Adiciona filtros ao título
    if filters:
        filter_str = " | ".join([f"{k}: {v}" for k, v in filters.items() if v])
        if filter_str:
            base_title += f" ({filter_str})"
    
    return base_title


@app.post("/generate_chart")
async def generate_chart(chart_data: dict):
    """Endpoint FLEXÍVEL para gerar gráficos com qualquer filtro."""
    try:
        # Importa bibliotecas
        import plotly.express as px
        import plotly.graph_objects as go
        import io
        import base64
        import pandas as pd
        import numpy as np
        import json
        
        print(f"📊 Gerando gráfico FLEXÍVEL com dados: {json.dumps(chart_data, indent=2)}")
        
        # Função para mapear nomes de colunas
        def _mapear_nome_coluna(nome):
            """Mapeia nomes comuns para nomes reais de colunas."""
            if not nome:
                return nome
            
            mapeamento = {
                'mes': 'mes',
                'mês': 'mes',
                'categorias': 'categoria',
                'categoria': 'categoria',
                'região': 'regiao',
                'regiao': 'regiao',
                'regiões': 'regiao',
                'produtos': 'produto',
                'produto': 'produto',
                'vendedores': 'vendedor',
                'vendedor': 'vendedor',
                'anos': 'ano',
                'ano': 'ano',
                'trimestres': 'trimestre',
                'trimestre': 'trimestre',
                'número': 'quantidade',
                'numero': 'quantidade',
                'quantidade': 'quantidade',
                'vendas': 'valor_total',
                'venda': 'valor_total',
                'valor': 'valor_total',
                'total': 'valor_total',
                'lucros': 'lucro',
                'lucro': 'lucro',
                'margens': 'margem_lucro',
                'margem': 'margem_lucro',
                'ticket': 'valor_total',
                'ticket médio': 'valor_total',
                'status': 'status',
                'forma_pagamento': 'forma_pagamento',
                'pagamento': 'forma_pagamento'
            }
            return mapeamento.get(nome.lower(), nome)
        
        # Usa o DataFrame global
        global df
        filtered_df = df.copy()
        
        # 1. FILTROS DINÂMICOS - Processa QUALQUER filtro que vier
        filters = chart_data.get("filters", {})
        
        if filters:
            print(f"🔍 Aplicando filtros: {filters}")
            
            for column, value in filters.items():
                column_mapped = _mapear_nome_coluna(column)
                
                if column_mapped in filtered_df.columns:
                    # Converte valor para o tipo correto da coluna
                    if filtered_df[column_mapped].dtype == 'int64' and isinstance(value, str):
                        try:
                            value = int(value)
                        except:
                            pass
                    elif filtered_df[column_mapped].dtype == 'float64' and isinstance(value, str):
                        try:
                            value = float(value)
                        except:
                            pass
                    
                    # Aplica filtro
                    if pd.notna(value):  # Só filtra se o valor não for None/NaN
                        if isinstance(value, list):
                            # Filtro com múltiplos valores (ex: categorias ["ELETRÔNICOS", "MODA"])
                            filtered_df = filtered_df[filtered_df[column_mapped].isin(value)]
                        else:
                            # Filtro com valor único
                            filtered_df = filtered_df[filtered_df[column_mapped] == value]
                else:
                    print(f"⚠️ Coluna '{column}' (mapeada para '{column_mapped}') não encontrada no dataset")
        
        # 2. FILTRO OBRIGATÓRIO: Apenas vendas CONCLUÍDAS (se a coluna existir)
        if 'status' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['status'] == 'CONCLUÍDA']
        
        # Verifica se sobrou dados após filtros
        if len(filtered_df) == 0:
            return {
                "error": "Nenhum dado encontrado com os filtros aplicados.",
                "success": False,
                "filters_applied": filters
            }
        
        # 3. PARÂMETROS DO GRÁFICO (com valores padrão inteligentes)
        chart_type = chart_data.get("chart_type", "bar")
        
        # Mapeia os nomes dos eixos
        x_axis_raw = chart_data.get("x_axis", "")
        y_axis_raw = chart_data.get("y_axis", "")
        
        x_axis = _mapear_nome_coluna(x_axis_raw)
        y_axis = _mapear_nome_coluna(y_axis_raw)
        
        # Se não foi especificado, sugere eixos inteligentes
        if not x_axis:
            x_axis = _sugerir_eixo_x(chart_type, filtered_df)
        
        if not y_axis:
            y_axis = _sugerir_eixo_y(chart_type, filtered_df)
        
        # Verifica se as colunas existem
        def _verificar_e_corrigir_coluna(coluna, df_ref, nome_tipo="eixo"):
            """Verifica se coluna existe e sugere alternativa se não."""
            if coluna in df_ref.columns:
                return coluna
            
            # Tenta encontrar coluna similar
            coluna_lower = coluna.lower()
            colunas_disponiveis = df_ref.columns.tolist()
            
            for col in colunas_disponiveis:
                col_lower = col.lower()
                if (coluna_lower in col_lower or col_lower in coluna_lower or 
                    coluna_lower.replace('_', '') == col_lower.replace('_', '')):
                    print(f"   🔄 Corrigido {nome_tipo}: '{coluna}' -> '{col}'")
                    return col
            
            # Se não encontrou, tenta mapeamento genérico
            mapeamento_generico = {
                'mes': ['month', 'mês', 'meses'],
                'categoria': ['category', 'categorias', 'cat'],
                'regiao': ['region', 'região', 'regioes'],
                'produto': ['product', 'produtos', 'prod'],
                'ano': ['year', 'anos'],
                'trimestre': ['quarter', 'quarters', 'trimestres'],
                'valor_total': ['total', 'valor', 'venda', 'sales', 'amount'],
                'quantidade': ['quantity', 'qty', 'qtd', 'numero', 'número'],
                'lucro': ['profit', 'lucros'],
                'margem_lucro': ['margin', 'margem', 'profit_margin']
            }
            
            for col_base, alternativas in mapeamento_generico.items():
                if coluna_lower in alternativas or any(alt in coluna_lower for alt in alternativas):
                    if col_base in df_ref.columns:
                        print(f"   🔄 Corrigido {nome_tipo} por mapeamento: '{coluna}' -> '{col_base}'")
                        return col_base
            
            # Se ainda não encontrou, retorna a primeira coluna disponível do tipo apropriado
            if nome_tipo == "eixo X":
                # Para eixo X, prioriza colunas categóricas
                categorical_cols = [c for c in df_ref.columns if df_ref[c].dtype == 'object']
                if categorical_cols:
                    return categorical_cols[0]
            
            # Para eixo Y ou fallback, retorna primeira coluna numérica
            numeric_cols = df_ref.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return numeric_cols[0]
            
            # Último recurso: primeira coluna
            return df_ref.columns[0]
        
        # Verifica e corrige eixos se necessário
        x_axis = _verificar_e_corrigir_coluna(x_axis, filtered_df, "eixo X")
        y_axis = _verificar_e_corrigir_coluna(y_axis, filtered_df, "eixo Y")
        
        # Agregação
        aggregation = chart_data.get("aggregation", "sum")
        
        # Se y_axis for "quantidade" e a pergunta pedir "número de vendas", usa contagem
        if y_axis == "quantidade" and aggregation == "count":
            # Para contagem de vendas, não precisamos de coluna y específica
            pass
        elif aggregation == "count" and y_axis not in filtered_df.columns:
            # Se for contagem e a coluna Y não existir, muda para contagem simples
            aggregation = "size"
            y_axis = "contagem"  # Placeholder
        
        # Título
        title = chart_data.get("title", _gerar_titulo_automatico(chart_type, x_axis, y_axis, filters))
        
        print(f"📐 Parâmetros finais: tipo={chart_type}, x={x_axis}, y={y_axis}, agregação={aggregation}")
        print(f"   🔍 Colunas disponíveis: {filtered_df.columns.tolist()}")
        
        # 4. PREPARAÇÃO DOS DADOS COM AGRAGAÇÃO DINÂMICA
        def _preparar_dados_para_grafico(df_local, x_col, y_col, chart_t, agg):
            """Prepara dados com agregação dinâmica com tratamento robusto de tipos."""
            
            print(f"🔍 Preparando dados: X='{x_col}', Y='{y_col}', tipo='{chart_t}', agregação='{agg}'")
            print(f"   🔍 Tipos: X={df_local[x_col].dtype if x_col in df_local.columns else 'N/A'}, "
                f"Y={df_local[y_col].dtype if y_col in df_local.columns else 'N/A'}")
            
            # Verifica se as colunas existem
            if x_col not in df_local.columns:
                raise ValueError(f"Coluna '{x_col}' não existe no dataset. Colunas disponíveis: {df_local.columns.tolist()}")
            
            # Para gráfico de dispersão - tratamento especial
            if chart_t == "scatter":
                print(f"   📊 Preparando dados para SCATTER plot")
                
                # Verifica se o eixo X existe
                if x_col not in df_local.columns:
                    numeric_cols = df_local.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        x_col = numeric_cols[0]
                        print(f"     ↳ Eixo X ajustado para: '{x_col}'")
                    else:
                        raise ValueError(f"Nenhuma coluna numérica disponível para scatter plot")
                
                # Verifica se o eixo Y existe e é apropriado
                if y_col not in df_local.columns:
                    # Tenta encontrar coluna numérica apropriada
                    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        # Prioriza 'quantidade' para "número de produtos vendidos"
                        if "quantidade" in numeric_cols:
                            y_col = "quantidade"
                        elif "valor_total" in numeric_cols:
                            y_col = "valor_total"
                        else:
                            y_col = numeric_cols[0]
                        print(f"     ↳ Eixo Y ajustado para: '{y_col}'")
                    else:
                        y_col = df_local.columns[1] if len(df_local.columns) > 1 else df_local.columns[0]
                
                # VERIFICAÇÃO CRÍTICA: Garante que as colunas sejam numéricas
                if df_local[x_col].dtype == 'object' or pd.api.types.is_string_dtype(df_local[x_col]):
                    print(f"     ⚠️ Coluna X '{x_col}' é textual, tentando converter...")
                    # Tenta converter para numérico se possível
                    try:
                        df_local[x_col] = pd.to_numeric(df_local[x_col], errors='coerce')
                        print(f"     ↳ Convertido '{x_col}' para numérico")
                    except:
                        # Se não conseguir converter, procura coluna numérica alternativa
                        numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            x_col = numeric_cols[0]
                            print(f"     ↳ Eixo X alterado para coluna numérica: '{x_col}'")
                        else:
                            raise ValueError(f"Coluna '{x_col}' é textual e não pode ser usada em scatter plot")
                
                if y_col in df_local.columns and (df_local[y_col].dtype == 'object' or pd.api.types.is_string_dtype(df_local[y_col])):
                    print(f"     ⚠️ Coluna Y '{y_col}' é textual, tentando converter...")
                    try:
                        df_local[y_col] = pd.to_numeric(df_local[y_col], errors='coerce')
                        print(f"     ↳ Convertido '{y_col}' para numérico")
                    except:
                        # Procura coluna numérica alternativa
                        numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            y_col = numeric_cols[0]
                            print(f"     ↳ Eixo Y alterado para coluna numérica: '{y_col}'")
                        else:
                            raise ValueError(f"Coluna '{y_col}' é textual e não pode ser usada em scatter plot")
                
                # Remove valores NaN após conversão
                result_df = df_local[[x_col, y_col]].dropna()
                
                # Verifica se sobrou dados suficientes
                if len(result_df) == 0:
                    raise ValueError(f"Nenhum dado numérico disponível após limpeza para scatter plot")
                
                print(f"     ✅ Dados scatter preparados: {len(result_df)} pontos")
                return result_df
            
            # Para gráfico de pizza - tratamento especial para categorias
            if chart_t == "pie":
                print(f"   🥧 Preparando dados para PIE chart")
                
                # Para pizza, geralmente agrupamos por categoria e somamos valores
                if y_col not in df_local.columns:
                    # Se não tem Y, conta as ocorrências
                    plot_df = df_local.groupby(x_col).size().reset_index(name='contagem')
                    y_col = 'contagem'
                else:
                    # Aplica agregação padrão para pizza
                    if agg == "sum":
                        plot_df = df_local.groupby(x_col)[y_col].sum().reset_index()
                    elif agg == "mean":
                        plot_df = df_local.groupby(x_col)[y_col].mean().reset_index()
                    elif agg == "count":
                        plot_df = df_local.groupby(x_col)[y_col].count().reset_index()
                    else:
                        plot_df = df_local.groupby(x_col)[y_col].sum().reset_index()
                
                # Ordena do maior para o menor para melhor visualização
                if pd.api.types.is_numeric_dtype(plot_df[plot_df.columns[1]]):
                    plot_df = plot_df.sort_values(by=plot_df.columns[1], ascending=False)
                
                # Limita a um número razoável de fatias (máximo 8)
                if len(plot_df) > 8:
                    print(f"     📋 Limitando pizza a 8 principais categorias")
                    top_data = plot_df.head(7)
                    others_sum = plot_df.iloc[7:][plot_df.columns[1]].sum()
                    others_row = {x_col: 'Outros', plot_df.columns[1]: others_sum}
                    plot_df = pd.concat([top_data, pd.DataFrame([others_row])], ignore_index=True)
                
                return plot_df
            
            # Para contagem de registros (não precisa de y_col específica)
            if agg == "size" or (agg == "count" and y_col not in df_local.columns):
                print(f"   📈 Preparando contagem de registros por '{x_col}'")
                plot_df = df_local.groupby(x_col).size().reset_index(name='contagem')
                print(f"     ✅ Resultado: {len(plot_df)} categorias")
                return plot_df
            
            # Para outros gráficos, verifica y_col
            if y_col not in df_local.columns:
                # Tenta encontrar coluna alternativa
                print(f"   ⚠️ Coluna Y '{y_col}' não encontrada, procurando alternativa...")
                
                # Para "número de produtos vendidos" ou similar, usa quantidade
                if any(term in y_col.lower() for term in ['número', 'numero', 'quantidade', 'contagem', 'produtos', 'vendas']):
                    if 'quantidade' in df_local.columns:
                        y_col = 'quantidade'
                        print(f"     ↳ Usando 'quantidade' como alternativa")
                    else:
                        # Conta registros como fallback
                        plot_df = df_local.groupby(x_col).size().reset_index(name='contagem')
                        y_col = 'contagem'
                        print(f"     ↳ Usando contagem de registros como alternativa")
                        return plot_df
                else:
                    # Tenta coluna numérica genérica
                    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        y_col = numeric_cols[0]
                        print(f"     ↳ Usando '{y_col}' (numérica) como alternativa")
                    else:
                        raise ValueError(f"Coluna '{y_col}' não existe e não há alternativas numéricas. "
                                    f"Colunas disponíveis: {df_local.columns.tolist()}")
            
            print(f"   📊 Aplicando agregação '{agg}' em '{y_col}' agrupado por '{x_col}'")
            
            # Aplica agregação
            if agg == "sum":
                plot_df = df_local.groupby(x_col)[y_col].sum().reset_index()
            elif agg == "mean":
                plot_df = df_local.groupby(x_col)[y_col].mean().reset_index()
            elif agg == "median":
                plot_df = df_local.groupby(x_col)[y_col].median().reset_index()
            elif agg == "count":
                plot_df = df_local.groupby(x_col)[y_col].count().reset_index()
            elif agg == "max":
                plot_df = df_local.groupby(x_col)[y_col].max().reset_index()
            elif agg == "min":
                plot_df = df_local.groupby(x_col)[y_col].min().reset_index()
            elif agg == "none":
                # Sem agregação - para alguns casos especiais
                plot_df = df_local[[x_col, y_col]].dropna()
            else:
                plot_df = df_local.groupby(x_col)[y_col].sum().reset_index()  # default
            
            # Ordena para melhor visualização
            if chart_t in ["bar", "line"] and len(plot_df) > 0 and pd.api.types.is_numeric_dtype(plot_df[plot_df.columns[1]]):
                plot_df = plot_df.sort_values(by=plot_df.columns[1], ascending=False)
                print(f"     📊 Dados ordenados por valor (maior para menor)")
            
            print(f"     ✅ Dados preparados: {len(plot_df)} linhas")
            return plot_df

        try:
            plot_df = _preparar_dados_para_grafico(
                filtered_df, x_axis, y_axis, chart_type, aggregation
            )
        except ValueError as e:
            print(f"❌ Erro ao preparar dados: {e}")
            
            # Tenta fallback inteligente baseado no tipo de gráfico
            try:
                print(f"   🔄 Tentando fallback inteligente...")
                
                if chart_type == "scatter":
                    # Fallback para scatter: usa valor_total vs quantidade
                    if "valor_total" in filtered_df.columns and "quantidade" in filtered_df.columns:
                        x_fallback = "valor_total"
                        y_fallback = "quantidade"
                        aggregation = "none"
                        print(f"     ↳ Fallback scatter: x={x_fallback}, y={y_fallback}")
                    else:
                        # Se não tem essas colunas, muda para gráfico de barras
                        chart_type = "bar"
                        x_fallback = "categoria" if "categoria" in filtered_df.columns else filtered_df.columns[0]
                        y_fallback = "valor_total" if "valor_total" in filtered_df.columns else filtered_df.select_dtypes(include=[np.number]).columns[0]
                        aggregation = "sum"
                        print(f"     ↳ Convertendo para barras: x={x_fallback}, y={y_fallback}")
                else:
                    # Fallback padrão para outros tipos
                    x_fallback = "categoria" if "categoria" in filtered_df.columns else filtered_df.columns[0]
                    y_fallback = "valor_total" if "valor_total" in filtered_df.columns else filtered_df.select_dtypes(include=[np.number]).columns[0]
                    print(f"     ↳ Fallback padrão: x={x_fallback}, y={y_fallback}")
                
                # Atualiza título para refletir as mudanças
                if chart_type == "scatter":
                    title = f"Dispersão: {y_fallback} vs {x_fallback}"
                else:
                    title = f"{chart_type.title()}: {y_fallback} por {x_fallback}"
                
                plot_df = _preparar_dados_para_grafico(
                    filtered_df, x_fallback, y_fallback, chart_type, aggregation
                )
                x_axis, y_axis = x_fallback, y_fallback
                
                print(f"   ✅ Fallback aplicado com sucesso")
                
            except Exception as fallback_err:
                print(f"❌ Erro no fallback: {fallback_err}")
                return {
                    "error": f"Erro ao preparar dados: {str(fallback_err)}",
                    "success": False,
                    "suggestion": "Tente usar termos mais específicos como 'valor_total por categoria' ou 'quantidade por mês'."
                }
        
        # 5. GERAÇÃO DO GRÁFICO (com fallback automático)
        def _criar_grafico_com_fallback(plot_df_local, original_df, chart_t, x_col, y_col, title_text):
            """Cria gráfico com fallback automático se um tipo falhar."""
            try:
                # Determina nome da coluna Y para exibição
                y_display = y_col
                if 'contagem' in plot_df_local.columns:
                    y_display = 'contagem'
                
                if chart_t == "bar":
                    # Tenta gráfico de barras agrupado se tiver subcategorias
                    if len(plot_df_local) > 15:  # Muitas categorias
                        plot_df_local = plot_df_local.head(10)  # Limita às top 10
                        title_text += " (Top 10)"
                    
                    fig = px.bar(
                        plot_df_local, x=x_col, y=y_display, title=title_text,
                        color=x_col if len(plot_df_local) < 8 else None,
                        text=y_display if len(plot_df_local) < 15 else None,
                        labels={x_col: x_col.replace('_', ' ').title(), 
                               y_display: y_display.replace('_', ' ').title()}
                    )
                    
                elif chart_t == "line":
                    fig = px.line(
                        plot_df_local, x=x_col, y=y_display, title=title_text,
                        markers=True, line_shape="spline",
                        labels={x_col: x_col.replace('_', ' ').title(), 
                               y_display: y_display.replace('_', ' ').title()}
                    )
                    
                elif chart_t == "pie":
                    # Limita para não ficar bagunçado
                    if len(plot_df_local) > 8:
                        plot_df_local = plot_df_local.sort_values(by=y_display, ascending=False)
                        others_sum = plot_df_local.iloc[7:][y_display].sum()
                        plot_df_local = plot_df_local.head(7)
                        plot_df_local = pd.concat([plot_df_local, pd.DataFrame({x_col: ['Outros'], y_display: [others_sum]})])
                        title_text += " (Top 7 + Outros)"
                    
                    fig = px.pie(
                        plot_df_local, names=x_col, values=y_display, title=title_text,
                        hole=0.3 if len(plot_df_local) > 4 else 0,
                        labels={x_col: x_col.replace('_', ' ').title(), 
                               y_display: y_display.replace('_', ' ').title()}
                    )
                    
                elif chart_t == "scatter":
                    fig = px.scatter(
                        original_df, x=x_col, y=y_display, title=title_text,
                        trendline="ols",
                        size=y_display if y_display in original_df.columns else None,
                        hover_data=original_df.columns.tolist(),
                        labels={x_col: x_col.replace('_', ' ').title(), 
                               y_display: y_display.replace('_', ' ').title()}
                    )
                    
                else:
                    # Fallback para gráfico de barras
                    fig = px.bar(plot_df_local, x=x_col, y=y_display, 
                                title=f"(Fallback) {title_text}",
                                labels={x_col: x_col.replace('_', ' ').title(), 
                                       y_display: y_display.replace('_', ' ').title()})
                    
                return fig
                
            except Exception as e:
                print(f"⚠️ Fallback no gráfico {chart_t}: {e}")
                # Fallback sempre para barras
                return px.bar(plot_df_local, x=x_col, y=y_display, 
                             title=f"(Simplificado) {title_text}",
                             labels={x_col: x_col.replace('_', ' ').title(), 
                                    y_display: y_display.replace('_', ' ').title()})
        
        fig = _criar_grafico_com_fallback(
            plot_df, filtered_df, chart_type, x_axis, 
            'contagem' if 'contagem' in plot_df.columns else y_axis, 
            title
        )
        
        # 6. FORMATAÇÃO INTELIGENTE
        def _aplicar_formatacao_inteligente(fig_local, x_col, y_col, chart_t):
            """Aplica formatação inteligente baseada nos eixos."""
            
            # Formatação monetária
            monetary_columns = ['valor', 'lucro', 'custo', 'preco', 'ticket', 'total', 'venda']
            y_display = y_col if y_col != 'contagem' else 'contagem'
            
            if any(money_word in y_display.lower() for money_word in monetary_columns):
                fig_local.update_yaxes(
                    tickprefix="R$ ",
                    tickformat=",.2f",
                    title_text=y_display.replace('_', ' ').title()
                )
            
            # Formatação percentual
            if 'percentual' in y_display.lower() or 'margem' in y_display.lower():
                fig_local.update_yaxes(
                    ticksuffix="%",
                    tickformat=".1%",
                    title_text=y_display.replace('_', ' ').title()
                )
            
            # Formatação para contagem
            if y_display == 'contagem':
                fig_local.update_yaxes(
                    title_text="Número de Vendas",
                    tickformat=",d"
                )
            
            # Ajusta layout baseado no tipo
            if chart_t == "bar" and len(fig_local.data[0].x) > 6:
                fig_local.update_layout(xaxis_tickangle=-45)
            
            if chart_t in ["line", "scatter"]:
                fig_local.update_layout(hovermode="x unified")
            
            # Melhora a legibilidade geral
            fig_local.update_layout(
                font=dict(size=12),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig_local
        
        fig = _aplicar_formatacao_inteligente(fig, x_axis, 
                                            'contagem' if 'contagem' in plot_df.columns else y_axis, 
                                            chart_type)
        
        # 7. CONVERSÃO PARA IMAGEM
        def _converter_grafico_para_imagem(fig_local):
            """Converte gráfico para imagem base64 com tratamento de erro."""
            import io
            import base64
            
            try:
                buf = io.BytesIO()
                fig_local.write_image(buf, format="png", width=1000, height=600, scale=2)
                buf.seek(0)
                return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            except Exception as e:
                print(f"⚠️ Erro ao converter gráfico: {e}")
                # Fallback: converte para HTML/JSON
                return {"plotly_json": fig_local.to_json(), "type": "interactive"}
        
        img_base64 = _converter_grafico_para_imagem(fig)
        
        # 8. GERA HTML PARA EMBED
        def _gerar_html_embed(img_b64, title_text, chart_t):
            """Gera HTML para embed no chat."""
            if isinstance(img_b64, dict):  # Se for JSON interativo
                return f'''
                <div class="plotly-chart" data-json='{img_b64["plotly_json"]}'>
                    <p><strong>{title_text}</strong> (Gráfico interativo - requer Plotly.js)</p>
                </div>
                '''
            
            return f'''
            <div class="chart-container">
                <h4 style="color: #4285f4; margin-bottom: 10px;">📊 {title_text}</h4>
                <img src="{img_b64}" 
                     style="max-width: 95%; border-radius: 10px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
                            border: 1px solid #e0e0e0; cursor: pointer;"
                     onclick="this.style.maxWidth=this.style.maxWidth==='95%'?'50%':'95%'"
                     title="Clique para ampliar/reduzir">
                <p style="color: #5f6368; font-size: 12px; margin-top: 5px;">
                    Tipo: {chart_t} | Clique na imagem para ajustar tamanho
                </p>
            </div>
            '''
        
        # 9. RETORNO RICO COM METADADOS
        return {
            "success": True,
            "type": "chart",
            "image": img_base64,
            "chart_type": chart_type,
            "data_points": len(plot_df),
            "filters_applied": filters,
            "dimensions": {
                "x_axis": x_axis,
                "y_axis": 'contagem' if 'contagem' in plot_df.columns else y_axis,
                "aggregation": aggregation
            },
            "data_preview": {
                "total_records": len(filtered_df),
                "x_categories": list(plot_df[x_axis].unique())[:5] if x_axis in plot_df.columns else [],
                "y_range": {
                    "min": float(plot_df[plot_df.columns[1]].min()) if (len(plot_df.columns) > 1 and 
                                                                    pd.api.types.is_numeric_dtype(plot_df[plot_df.columns[1]])) else 0,
                    "max": float(plot_df[plot_df.columns[1]].max()) if (len(plot_df.columns) > 1 and 
                                                                    pd.api.types.is_numeric_dtype(plot_df[plot_df.columns[1]])) else 0,
                    "sum": float(plot_df[plot_df.columns[1]].sum()) if (len(plot_df.columns) > 1 and 
                                                                    pd.api.types.is_numeric_dtype(plot_df[plot_df.columns[1]])) else 0
                }
            },
            "html_embed": _gerar_html_embed(img_base64, title, chart_type)
        }
        
    except Exception as e:
        print(f"❌ Erro no gráfico flexível: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Erro ao gerar gráfico: {str(e)}",
            "success": False,
            "suggestion": "Tente simplificar sua solicitação ou usar filtros diferentes."
        }

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """Endpoint principal para consultas de análise."""
    if analista is None:
        return {
            "response": "❌ **Erro de Inicialização**\n\nServidor não conseguiu carregar os dados.",
            "success": False,
            "type": "error"
        }
    
    try:
        question = request.message
        
        print(f"📝 Recebido: {question}")
        
        # ========== DETECÇÃO DE GRÁFICO ==========
        # 1. Extrai entidades E parâmetros de gráfico
        entities, chart_params = analista._extrair_entidades(question)
        
        print(f"🔍 Entidades extraídas: {entities}")
        print(f"📊 Parâmetros gráfico: {chart_params}")
        
        # 2. SÓ processa como gráfico se:
        #    a) Tem parâmetros de gráfico
        #    b) Tem tipo de gráfico definido
        #    c) A pergunta EXPLICITAMENTE pede gráfico
        question_lower = question.lower()
        explicit_chart_words = ['gráfico', 'grafico', 'chart', 'barras', 'linha', 'pizza', 'torta', 'dispersão']
        
        has_explicit_chart_request = any(word in question_lower for word in explicit_chart_words)
        has_chart_params = chart_params and chart_params.get("chart_type")
        
        if has_chart_params and has_explicit_chart_request:
            print(f"🎯 Processando como GRÁFICO (explícito)...")
            
            try:
                # Garante que temos os parâmetros mínimos
                if not chart_params.get("x_axis"):
                    # Sugere eixo X inteligente com mapeamento correto
                    column_mapping = {
                        'mês': 'mes', 'mes': 'mes', 'mês': 'mes',
                        'categorias': 'categoria', 'categoria': 'categoria',
                        'região': 'regiao', 'regiao': 'regiao',
                        'produtos': 'produto', 'produto': 'produto',
                        'vendedores': 'vendedor', 'vendedor': 'vendedor',
                        'anos': 'ano', 'ano': 'ano',
                        'trimestres': 'trimestre', 'trimestre': 'trimestre'
                    }
                    
                    for word, column in column_mapping.items():
                        if word in question_lower:
                            chart_params["x_axis"] = column
                            print(f"     ↳ Eixo X detectado: '{word}' -> '{column}'")
                            break
                    
                    if not chart_params.get("x_axis"):
                        chart_params["x_axis"] = "mes"  # padrão
                
                # Verifica se a coluna existe no dataset
                if chart_params.get("x_axis") not in df.columns:
                    # Tenta encontrar coluna similar
                    available_columns = df.columns.tolist()
                    x_lower = chart_params.get("x_axis", "").lower()
                    
                    for col in available_columns:
                        if x_lower in col.lower() or col.lower() in x_lower:
                            chart_params["x_axis"] = col
                            print(f"     ↳ Corrigido eixo X: '{x_lower}' -> '{col}'")
                            break
                
                if not chart_params.get("y_axis"):
                    # Sugere eixo Y inteligente
                    if 'quantidade' in question_lower or 'numero' in question_lower or 'número' in question_lower:
                        chart_params["y_axis"] = "quantidade"
                    elif 'lucro' in question_lower:
                        chart_params["y_axis"] = "lucro"
                    elif 'margem' in question_lower:
                        chart_params["y_axis"] = "margem_lucro"
                    elif 'venda' in question_lower and ('número' in question_lower or 'quantidade' in question_lower):
                        chart_params["y_axis"] = "quantidade"
                        chart_params["aggregation"] = "count"
                    else:
                        chart_params["y_axis"] = "valor_total"
                
                # Verifica se a coluna Y existe
                if chart_params.get("y_axis") not in df.columns:
                    available_columns = df.columns.tolist()
                    y_lower = chart_params.get("y_axis", "").lower()
                    
                    for col in available_columns:
                        if y_lower in col.lower() or col.lower() in y_lower:
                            chart_params["y_axis"] = col
                            print(f"     ↳ Corrigido eixo Y: '{y_lower}' -> '{col}'")
                            break
                
                # Detecção de agregação
                if "médio" in question_lower or "média" in question_lower:
                    chart_params["aggregation"] = "mean"
                elif "total" in question_lower or "soma" in question_lower:
                    chart_params["aggregation"] = "sum"
                elif "contagem" in question_lower or "quantos" in question_lower or "número" in question_lower:
                    chart_params["aggregation"] = "count"
                elif not chart_params.get("aggregation"):
                    chart_params["aggregation"] = "sum"
                
                # Gera título se não tiver
                if not chart_params.get("title"):
                    tipo_nome = {
                        "bar": "Gráfico de Barras",
                        "line": "Gráfico de Linha",
                        "pie": "Gráfico de Pizza",
                        "scatter": "Gráfico de Dispersão"
                    }.get(chart_params["chart_type"], "Gráfico")
                    
                    y_label = chart_params.get("y_axis", "valor")
                    if chart_params.get("aggregation") == "count":
                        y_label = "Número de Vendas"
                    
                    chart_params["title"] = f"{tipo_nome}: {y_label} por {chart_params.get('x_axis', 'categoria')}"
                    
                    # Adiciona filtros ao título
                    if chart_params.get("filters"):
                        filtros_str = []
                        for k, v in chart_params["filters"].items():
                            if isinstance(v, list):
                                filtros_str.append(f"{k}: {', '.join(v[:2])}{'...' if len(v) > 2 else ''}")
                            else:
                                filtros_str.append(f"{k}: {v}")
                        
                        if filtros_str:
                            chart_params["title"] += f" ({', '.join(filtros_str)})"
                
                print(f"🎨 Parâmetros finais do gráfico: {chart_params}")
                
                # 3. CHAMA O ENDPOINT DE GRÁFICO
                chart_response = await generate_chart(chart_params)
                
                if chart_response.get("success"):
                    # Retorna HTML com a imagem do gráfico
                    img_base64 = chart_response["image"]
                    
                    # No endpoint /chat, dentro do if chart_response.get("success"):
                    html_response = f'''
                    <div style="
                        background: white; 
                        border-radius: 12px; 
                        border: 1px solid #e0e0e0; 
                        padding: 12px; 
                        margin: 12px 0;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                        max-width: 100%;
                        overflow: hidden;
                        box-sizing: border-box;
                    ">
                        <div style="
                            display: flex; 
                            align-items: center; 
                            gap: 8px; 
                            margin-bottom: 10px;
                            color: #4285f4;
                        ">
                            <span style="font-size: 18px;">📊</span>
                            <h4 style="margin: 0; font-size: 14px; font-weight: 500; line-height: 1.3;">
                                {chart_params.get('title', 'Gráfico Gerado')}
                            </h4>
                        </div>
                        
                        <div style="
                            text-align: center; 
                            max-width: 100%; 
                            overflow-x: auto;
                            padding: 5px;
                            background: #f9f9f9;
                            border-radius: 8px;
                            border: 1px solid #f0f0f0;
                        ">
                            <img src="{img_base64}" 
                                id="chartImg_{chart_response.get('chart_type', 'default')}_{id(chart_response)}"
                                style="
                                    max-width: 95%;
                                    max-height: 280px;
                                    width: auto;
                                    height: auto;
                                    border-radius: 6px; 
                                    cursor: pointer;
                                    transition: all 0.2s ease;
                                    display: block;
                                    margin: 0 auto;
                                "
                                onclick="toggleChartSize(this.id)"
                                title="Clique para ampliar/reduzir"
                                alt="Gráfico gerado"
                            >
                        </div>
                        
                        <div style="
                            margin-top: 10px; 
                            padding-top: 10px; 
                            border-top: 1px solid #f0f0f0;
                            font-size: 11px;
                            color: #5f6368;
                            line-height: 1.4;
                        ">
                            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 6px;">
                                <span><strong style="color: #5f6368;">Tipo:</strong> {chart_response.get('chart_type', 'N/A')}</span>
                                <span><strong style="color: #5f6368;">Pontos:</strong> {chart_response.get('data_points', 0)}</span>
                                <span><strong style="color: #5f6368;">Eixo X:</strong> {chart_params.get('x_axis', 'N/A')}</span>
                            </div>
                            <p style="margin: 0; font-size: 10px; color: #9aa0a6;">
                                <em>💡 Clique no gráfico para ampliar/reduzir</em>
                            </p>
                        </div>
                    </div>

                    <script>
                    function toggleChartSize(imgId) {{
                        var img = document.getElementById(imgId);
                        if (!img) return;
                        
                        if (img.style.maxWidth === '100%' || img.style.maxWidth === '') {{
                            img.style.maxWidth = '100%';
                            img.style.maxHeight = '500px';
                            img.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                            img.style.zIndex = '1000';
                            img.style.position = 'relative';
                            img.style.backgroundColor = 'white';
                        }} else {{
                            img.style.maxWidth = '95%';
                            img.style.maxHeight = '280px';
                            img.style.boxShadow = 'none';
                            img.style.zIndex = '1';
                            img.style.position = 'static';
                        }}
                    }}
                    </script>
                    '''
                    
                    return {
                        "response": html_response,
                        "success": True,
                        "type": "chart",
                        "metadata": {
                            "chart_type": chart_response.get("chart_type"),
                            "data_points": chart_response.get("data_points"),
                            "dimensions": chart_response.get("dimensions", {})
                        }
                    }
                else:
                    # Se falhou o gráfico, cai para análise normal com mensagem
                    error_msg = chart_response.get("error", "Erro desconhecido")
                    print(f"⚠️ Falha no gráfico, caindo para análise normal: {error_msg}")
                    
                    # Adiciona mensagem sobre o erro do gráfico
                    fallback_msg = f"📊 **Não foi possível gerar o gráfico solicitado**\n\n*Motivo: {error_msg}*\n\n*Processando como análise de dados normal...*"
                    
                    # Continua com análise normal
                    return await _processar_analise_normal(question, fallback_msg)
                    
            except Exception as chart_error:
                print(f"⚠️ Erro no processamento de gráfico: {chart_error}")
                import traceback
                traceback.print_exc()
                # Cai para análise normal
                return await _processar_analise_normal(question, f"⚠️ **Erro no gráfico, processando como análise normal...**")
        
        # ========== ANÁLISE NORMAL (NÃO É GRÁFICO ou não é explícito) ==========
        print(f"🔢 Processando como ANÁLISE DE DADOS...")
        return await _processar_analise_normal(question)
        
    except Exception as e:
        print(f"❌ Erro geral em /chat: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"❌ **Erro no processamento**\n\n`{str(e)[:200]}...`",
            "success": False,
            "type": "error"
        }
    
    finally:
        # Log automático de custos após cada consulta
        try:
            from api_cost_tracker import cost_tracker
            from datetime import datetime
            
            # ESTIMAÇÃO MELHORADA DE TOKENS
            question_words = len(question.split())
            prompt_tokens = max(int(question_words * 1.3), 50)
            
            # Estimativa conservadora para resposta
            completion_tokens = 300  # Estimativa média para análise
            
            cost = cost_tracker.track_call(
                model="gpt-4o",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={
                    "question": question[:100],
                    "function": "handle_chat",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            today = cost_tracker.get_daily_summary()
            print(f"💰 [CHAT] Custo: ${cost:.6f} | Total hoje: ${today['cost']:.6f}")
            print(f"   📝 Tokens: {prompt_tokens}+{completion_tokens} = {prompt_tokens + completion_tokens}")
            
        except ImportError:
            print("⚠️ Monitor de custos não disponível")
        except Exception as e:
            print(f"⚠️ Erro ao registrar custos: {e}")

# Função auxiliar para processar análise normal
async def _processar_analise_normal(question, prefix_msg=""):
    """Processa consulta normal de análise de dados."""
    from datetime import datetime
    try:
        # Verifica se é uma saudação ou pergunta não relacionada
        greetings = ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'hello', 'hi']
        if any(greet in question.lower() for greet in greetings):
            return {
                "response": "👋 **Olá!** Sou o SalesInsight AI, seu assistente para análise de dados de vendas.<br><br>Você pode me perguntar sobre:<br>• Vendas por período<br>• Lucros e margens<br>• Comparativos entre regiões<br>• Top produtos<br>• Gráficos e visualizações<br><br>Experimente usar os exemplos rápidos acima!",
                "success": True,
                "type": "greeting"
            }
        
        resposta, pandas_code_str = analista.consultar(question)
        print(f"🤖 Resposta gerada pelo analista")
        print(f"🤖 Código pandas gerado: {pandas_code_str[:200]}...")
        
        # DETECTA SE É CÓDIGO DE VISUALIZAÇÃO (matplotlib, seaborn, plotly, etc.)
        def is_visualization_code(code):
            """Verifica se o código é de visualização."""
            visualization_keywords = [
                'import matplotlib', 'plt.', 'matplotlib.pyplot',
                'import seaborn', 'sns.', 'seaborn',
                'import plotly', 'px.', 'plotly.', 'fig =', '.show()',
                '.plot(', '.hist(', '.scatter(', '.bar(', '.pie(',
                'plt.title', 'plt.xlabel', 'plt.ylabel', 'plt.legend',
                'sns.barplot', 'sns.lineplot', 'sns.scatterplot',
                'px.bar', 'px.line', 'px.scatter', 'px.pie'
            ]
            code_lower = code.lower()
            return any(keyword in code_lower for keyword in visualization_keywords)
        
        # SE FOR CÓDIGO DE VISUALIZAÇÃO, TENTA EXECUTAR
        if is_visualization_code(pandas_code_str):
            print(f"🎨 Detectado código de visualização, tentando executar...")
            
            img_base64 = executar_codigo_visualizacao(pandas_code_str, question)
            
            if img_base64:
                # Extrai título do gráfico se possível
                title = "Gráfico Gerado"
                lines = pandas_code_str.split('\n')
                for line in lines:
                    if 'plt.title(' in line or '.title(' in line:
                        if "'" in line:
                            title = line.split("'")[1] if len(line.split("'")) > 1 else "Gráfico"
                        elif '"' in line:
                            title = line.split('"')[1] if len(line.split('"')) > 1 else "Gráfico"
                        break
                
                # HTML do gráfico
                html_response = f'''
                <div style="
                    background: white; 
                    border-radius: 12px; 
                    border: 1px solid #e0e0e0; 
                    padding: 12px; 
                    margin: 12px 0;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                    max-width: 100%;
                ">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; color: #4285f4;">
                        <span style="font-size: 18px;">📊</span>
                        <h4 style="margin: 0; font-size: 14px; font-weight: 500;">{title}</h4>
                    </div>
                    
                    <div style="text-align: center; background: #f9f9fa; padding: 10px; border-radius: 8px;">
                        <img src="{img_base64}" 
                             id="chart_{id(img_base64)}"
                             style="
                                max-width: 95%; 
                                max-height: 300px; 
                                border-radius: 6px; 
                                cursor: pointer;
                                transition: all 0.2s;
                             "
                             onclick="
                                var img = this;
                                if (img.style.maxHeight === '500px') {{
                                    img.style.maxHeight = '300px';
                                    img.style.maxWidth = '95%';
                                }} else {{
                                    img.style.maxHeight = '500px';
                                    img.style.maxWidth = '100%';
                                }}
                             "
                             title="Clique para ampliar/reduzir">
                    </div>
                    
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #f0f0f0;">
                        <div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #5f6368;">
                            <span><strong>Tipo:</strong> Visualização</span>
                            <span><strong>Biblioteca:</strong> {"Matplotlib" if 'plt.' in pandas_code_str.lower() else "Seaborn" if 'sns.' in pandas_code_str.lower() else "Plotly" if 'plotly' in pandas_code_str.lower() else "Pandas"}</span>
                        </div>
                        <p style="margin: 5px 0 0; font-size: 10px; color: #9aa0a6;">
                            <em>💡 Gráfico gerado automaticamente a partir do código Python</em>
                        </p>
                    </div>
                </div>
                '''
                
                # Junta gráfico + resposta + código
                resposta_completa = f"{prefix_msg}<br>{html_response}<br><br>{resposta}"
                
                # Adiciona o código fonte (FORMATAÇÃO MELHORADA)
                if pandas_code_str and len(pandas_code_str.strip()) > 0:
                    # Limpa e formata o código
                    clean_code = pandas_code_str.strip()
                    
                    # Remove marcações de código se existirem
                    if '```python' in clean_code:
                        code_start = clean_code.find('```python') + 9
                        code_end = clean_code.find('```', code_start)
                        clean_code = clean_code[code_start:code_end].strip()
                    elif '```' in clean_code:
                        code_start = clean_code.find('```') + 3
                        code_end = clean_code.find('```', code_start)
                        clean_code = clean_code[code_start:code_end].strip()
                    
                    # Remove avisos de depreciação do código
                    if 'applymap' in clean_code and 'deprecated' not in clean_code:
                        # Adiciona warning suppression
                        clean_code = "import warnings\nwarnings.filterwarnings('ignore', message='DataFrame.applymap')\n" + clean_code
                        # Substitui applymap por map
                        clean_code = clean_code.replace('.applymap(', '.map(')
                    
                    # Adiciona código formatado
                    resposta_completa += f'''
                    <br><br>
                    <div style="
                        background: #f8f9fa; 
                        padding: 10px; 
                        border-radius: 6px; 
                        font-size: 11px; 
                        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                        overflow-x: auto;
                        border-left: 4px solid #4285f4;
                        margin-top: 15px;
                    ">
                        <div style="color: #5f6368; margin-bottom: 8px; font-size: 10px;">
                            <i class="fas fa-code"></i> Código gerado:
                        </div>
                        <code style="color: #202124; white-space: pre-wrap;">{clean_code[:800]}{'...' if len(clean_code) > 800 else ''}</code>
                    </div>
                    '''
                
                return {
                    "response": resposta_completa,
                    "success": True,
                    "type": "chart_visualization"
                }
            else:
                # Se falhou em gerar gráfico, cai para análise normal
                print(f"⚠️ Falha ao gerar visualização, processando como análise normal...")
                prefix_msg = f"{prefix_msg}<br>⚠️ <em>Não foi possível gerar a visualização solicitada. Mostrando análise de dados...</em><br>" if not prefix_msg else f"{prefix_msg}<br>⚠️ <em>Não foi possível gerar a visualização solicitada. Mostrando análise de dados...</em><br>"
        
        # SE NÃO FOR VISUALIZAÇÃO OU FALHOU, PROCESSA NORMALMENTE
        resultado_completo = executar_e_capturar_resultado(pandas_code_str, question)
        
        print(f"📋 Resultado completo obtido (tipo: {type(resultado_completo)})")
        
        # Formata resultado usando a função centralizada
        if resultado_completo is not None:
            print(f"🎨 Chamando formatar_resultado_pandas...")
            resposta_final = formatar_resultado_pandas(resultado_completo, question)
            print(f"✅ Resposta formatada (tamanho: {len(resposta_final) if resposta_final else 0})")
            
            # Se a formatação retornou vazio, usa a resposta do analista
            if not resposta_final or resposta_final == "":
                print(f"⚠️ Formatação retornou vazio, usando resposta original")
                resposta_final = resposta
        else:
            # Se não conseguiu executar o código, usa a resposta do analista
            resposta_final = resposta
            print(f"⚠️ resultado_completo é None, usando resposta original")
        
        # Se a resposta final ainda não é HTML (texto simples), converte
        if resposta_final and not ('<' in resposta_final and '>' in resposta_final):
            print(f"🔄 Convertendo texto simples para HTML básico")
            resposta_final = resposta_final.replace('\n', '<br>')
        
        # Limpa espaços em excesso da resposta final
        resposta_final = resposta_final.replace('\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
        
        # Adiciona prefixo se houver
        if prefix_msg:
            resposta_final = f"{prefix_msg}<br><br>{resposta_final}"
        
        # Prepara resposta final - NÃO ESCAPA HTML se já contém tags
        if '<' in resposta_final and '>' in resposta_final:
            # Já é HTML, não escapa
            resposta_completa = resposta_final
        else:
            # É texto simples, aplica formatação básica
            import html
            resposta_completa = html.escape(resposta_final)
            resposta_completa = resposta_completa.replace('\n', '<br>')
        
        # Adiciona o código fonte (FORMATAÇÃO MELHORADA)
        if pandas_code_str and len(pandas_code_str.strip()) > 0:
            # Limpa e formata o código
            clean_code = pandas_code_str.strip()
            
            # Remove marcações de código se existirem
            if '```python' in clean_code:
                code_start = clean_code.find('```python') + 9
                code_end = clean_code.find('```', code_start)
                clean_code = clean_code[code_start:code_end].strip()
            elif '```' in clean_code:
                code_start = clean_code.find('```') + 3
                code_end = clean_code.find('```', code_start)
                clean_code = clean_code[code_start:code_end].strip()
            
            # Remove avisos de depreciação do código
            if 'applymap' in clean_code and 'deprecated' not in clean_code:
                # Adiciona warning suppression
                clean_code = "import warnings\nwarnings.filterwarnings('ignore', message='DataFrame.applymap')\n" + clean_code
                # Substitui applymap por map
                clean_code = clean_code.replace('.applymap(', '.map(')
            
            # Adiciona código formatado
            resposta_completa += f'''
            <br><br>
            <div style="
                background: #f8f9fa; 
                padding: 10px; 
                border-radius: 6px; 
                font-size: 11px; 
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                overflow-x: auto;
                border-left: 4px solid #4285f4;
                margin-top: 15px;
            ">
                <div style="color: #5f6368; margin-bottom: 8px; font-size: 10px;">
                    <i class="fas fa-code"></i> Código gerado:
                </div>
                <code style="color: #202124; white-space: pre-wrap;">{clean_code[:800]}{'...' if len(clean_code) > 800 else ''}</code>
            </div>
            '''
        
        # Se a resposta contém HTML, garanta que está corretamente formatada
        if '<table' in resposta_completa or '<br>' in resposta_completa or '<div' in resposta_completa:
            # Não escapa - já é HTML válido
            pass
        else:
            # Se for texto simples, formata com quebras de linha
            resposta_completa = resposta_completa.replace('\n', '<br>')

        # Log de custos após análise completa
        try:
            from api_cost_tracker import cost_tracker
            from datetime import datetime
            
            # ESTIMAÇÃO MELHORADA DE TOKENS
            # Baseada no tamanho real da conversa
            question_words = len(question.split())
            response_words = len(str(resposta_completa).split()) if resposta_completa else 0
            
            # Conversão aproximada: 1 palavra ≈ 1.3 tokens em português
            prompt_tokens = int(question_words * 1.3)
            completion_tokens = int(response_words * 1.3)
            
            # Mínimo para garantir que algo seja registrado
            prompt_tokens = max(prompt_tokens, 50)
            completion_tokens = max(completion_tokens, 100)
            
            # Registra a chamada
            cost = cost_tracker.track_call(
                model="gpt-4o",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={
                    "question": question[:100],
                    "question_length": len(question),
                    "response_length": len(str(resposta_completa)) if resposta_completa else 0,
                    "function": "_processar_analise_normal",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Mostra no log
            today = cost_tracker.get_daily_summary()
            print(f"💰 [ANÁLISE COMPLETA] Custo: ${cost:.6f} | Total hoje: ${today['cost']:.6f}")
            print(f"   📝 Tokens: {prompt_tokens}+{completion_tokens} = {prompt_tokens + completion_tokens}")
            
        except Exception as e:
            print(f"⚠️ Erro ao registrar custos: {e}")

        return {
            "response": resposta_completa,
            "success": True,
            "type": "data"
        }
        
    except Exception as e:
        print(f"❌ Erro na análise normal: {e}")
        import traceback
        traceback.print_exc()

        # Log de custos mesmo com erro
        try:
            from api_cost_tracker import cost_tracker
            from datetime import datetime
            
            # Registra pelo menos a pergunta (mesmo com erro)
            question_words = len(question.split())
            prompt_tokens = max(int(question_words * 1.3), 50)
            
            cost_tracker.track_call(
                model="gpt-4o",
                prompt_tokens=prompt_tokens,
                completion_tokens=100,  # Estimativa mínima
                metadata={
                    "question": question[:100],
                    "error": str(e)[:200],
                    "function": "_processar_analise_normal_error",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            today = cost_tracker.get_daily_summary()
            print(f"💰 [ANÁLISE-ERRO] Registrado custo mínimo: ${today['cost']:.6f}")
            
        except Exception as cost_error:
            print(f"⚠️ Erro ao registrar custos de erro: {cost_error}")
        # -----------------------------------------------------------------
        return {
            "response": f"❌ **Erro na análise**<br><br>`{str(e)[:200]}`<br><br>Tente reformular sua pergunta ou usar termos mais específicos.",
            "success": False,
            "type": "error"
        }
# Para Vercel Functions
handler = app