# Lógica principal do analista de vendas
import os
import pandas as pd
import ast
import json
import time
from dotenv import load_dotenv

# LlamaIndex Core
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import set_global_handler

# Vector Store
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# LangChain
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# ================== MONITOR DE CUSTOS ==================
# Importar o monitor de custos
try:
    from api_cost_tracker import cost_tracker, track_openai_cost
    COST_TRACKING_ENABLED = True
    print("💰 Monitor de custos ativado!")
except ImportError:
    COST_TRACKING_ENABLED = False
    print("⚠️ Monitor de custos não disponível")

# Configuração
load_dotenv()
set_global_handler("simple")

CHROMADB_PATH = os.getenv('CHROMADB_PATH', './chromadb_sales')

# ================== DECORADOR PERSONALIZADO ==================
def track_llama_call(func):
    """
    Decorador personalizado para rastrear chamadas LlamaIndex/OpenAI.
    Este decorador captura os tokens usados nas consultas.
    """
    def wrapper(self, *args, **kwargs):
        if not COST_TRACKING_ENABLED:
            return func(self, *args, **kwargs)
        
        # Extrai a pergunta (primeiro argumento)
        question = args[0] if len(args) > 0 else kwargs.get('question', '')
        
        try:
            # Executa a função original
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            
            # Tenta extrair informações de uso
            if isinstance(result, tuple) and len(result) >= 2:
                resposta, codigo_pandas = result
                
                # ESTIMATIVA DE TOKENS (aproximada)
                # Regra geral: 1 token ≈ 4 caracteres em inglês, 2-3 em português
                prompt_tokens = len(question) // 2  # Estimativa conservadora
                
                # Se temos resposta do analista, estima tokens da resposta
                if resposta:
                    completion_tokens = len(resposta) // 2
                else:
                    completion_tokens = 500  # Estimativa padrão
                
                # Registra a chamada
                cost_tracker.track_call(
                    model="gpt-4o",  # Modelo usado no seu projeto
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    metadata={
                        "function": func.__name__,
                        "question": question[:100],  # Primeiros 100 chars
                        "execution_time": end_time - start_time,
                        "response_length": len(resposta) if resposta else 0
                    }
                )
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erro no monitor de custos: {e}")
            return func(self, *args, **kwargs)
    
    return wrapper

class SalesDataAnalyst:
    
    # Contexto para extração de entidades (ATUALIZADO)
    CONTEXT_EXTRACAO = """
    CONTEXTO DA CONVERSA ANTERIOR:
    {historico_conversa}

    EXTRACTION RULES FOR SALES DATA ANALYSIS:

    1. OUTPUT FORMAT: Return ONLY a VALID JSON with this exact structure:
    {
        "entities": [], 
        "chart_params": {}
    }

    2. ENTITIES EXTRACTION:
    - Extract only sales-related entities: years, months, categories, regions, metrics
    - Convert years to uppercase: "2025" → "2025"
    - Keep entities as simple strings, not lists or complex objects

    3. CHART PARAMETERS (ONLY when user explicitly requests visualization):
    - User must use words: "gráfico", "chart", "visualização", "barras", "linha", "pizza", "dispersão"
    - chart_type: "bar" (barras), "line" (linha), "pie" (pizza), "scatter" (dispersão)
    - x_axis: column name from dataset (e.g., "mes", "categoria", "regiao")
    - y_axis: column name from dataset (e.g., "valor_total", "quantidade", "lucro")
    - title: concise Portuguese title based on request
    - filters: {} object with ACTUAL filters, not placeholder values

    4. FILTERS RULES (CRITICAL):
    - WHEN USER MENTIONS A SPECIFIC YEAR: Add {"ano": YEAR_NUMBER}
        Example: "em 2025" → {"ano": 2025}
        Example: "para 2023" → {"ano": 2023}
    
    - WHEN USER MENTIONS SPECIFIC CATEGORIES: Add {"categoria": ["CAT1", "CAT2"]}
        Example: "eletrônicos e moda" → {"categoria": ["ELETRÔNICOS", "MODA"]}
    
    - DO NOT ADD "categorias": "todas" EVER unless user explicitly says "todas as categorias"
    - DO NOT ADD "regioes": "todas" EVER
    - Filters should contain ONLY what user explicitly mentioned

    5. CHART TYPE DETECTION:
    - "barras", "colunas", "bar chart" → "bar"
    - "linha", "tendência", "evolução" → "line"  
    - "pizza", "torta", "proporção", "percentual" → "pie"
    - "dispersão", "correlação", "scatter" → "scatter"

    6. AXIS DETECTION:
    - "por mês", "mensal", "ao longo dos meses" → x_axis: "mes"
    - "por categoria", "cada categoria" → x_axis: "categoria"
    - "por região", "por estado" → x_axis: "regiao"
    - "número de", "quantidade de" → y_axis: "quantidade"
    - "valor", "vendas", "total" → y_axis: "valor_total"
    - "lucro" → y_axis: "lucro"
    - "margem" → y_axis: "margem_lucro"

    7. WHEN NO CHART IS REQUESTED:
    - If user asks for data, numbers, analysis without visualization words
    - Set chart_params to empty object: {}

    8. EXAMPLES FOR COMMON QUERIES:

    Query: "Gráfico de barras da quantidade de vendas por mês em 2025"
    Output: {
        "entities": ["2025"],
        "chart_params": {
        "chart_type": "bar",
        "x_axis": "mes",
        "y_axis": "quantidade",
        "title": "Quantidade de Vendas por Mês em 2025",
        "filters": {"ano": 2025}
        }
    }

    Query: "Linha do valor total por categoria ao longo de 2024"
    Output: {
        "entities": ["2024"],
        "chart_params": {
        "chart_type": "line",
        "x_axis": "mes",
        "y_axis": "valor_total",
        "title": "Valor Total por Categoria em 2024",
        "filters": {"ano": 2024}
        }
    }

    Query: "Margem de lucro por categoria em 2023"
    Output: {
        "entities": ["2023"],
        "chart_params": {}
    }

    Query: "Top 5 produtos mais vendidos"
    Output: {
        "entities": [],
        "chart_params": {}
    }

    9. ERROR PREVENTION:
    - If uncertain, return empty: {"entities": [], "chart_params": {}}
    - Validate JSON structure before returning
    - chart_params must be {} not null or undefined

    USER QUERY TO ANALYZE: {input}
    """
    
    # Instruções para geração de código Pandas (ATUALIZADO)
    CONTEXT_PANDAS_INSTRUCOES = """
    You are a Pandas code generator for sales data analysis (DataFrame df).
    Your ONLY output must be the final executable Python expression.
    
    CONTEXT FROM PREVIOUS CONVERSATION:
    {historico_conversa}
    
    DATASET STRUCTURE:
    - id_venda, data, ano, mes, trimestre, dia_semana
    - categoria, produto, regiao, estado, cidade
    - vendedor, cliente, quantidade, preco_unitario
    - desconto_percentual, valor_total, forma_pagamento
    - custo_unitario, custo_total, lucro, margem_lucro, status
    
    BUSINESS CONTEXT:
    {contexto_rag}
    
    USER QUERY: {user_query}
    
    IMPORTANT: Consider previous questions and answers when generating code.
    If user asks follow-up questions (like "show top 5" after "total sales"),
    maintain context from previous analysis.
    
    CRITICAL RULES FOR CALCULATIONS:
    1. For valid sales only: df[df['status'] == 'CONCLUÍDA']
    2. Total sales = df['valor_total'].sum()
    3. Average ticket = df['valor_total'].mean()
    4. Profit margin = (df['lucro'].sum() / df['valor_total'].sum()) * 100
    5. Use .groupby() for aggregations
    6. Format monetary values with 2 decimals
    7. FOR CHARTS: Generate only the plot() expression without variable assignment
    Example: df[...].plot(kind='bar')  # CORRECT
    Not: chart = df[...].plot(kind='bar')  # WRONG
    8. For year comparison with growth percentage: Use .T (transpose) so years are in columns, then calculate: ((year2 - year1) / year1 * 100)
    
    **SPECIFIC PATTERN FOR COMPARISON TABLES:**
    9. For comparison tables between years, use this EXACT pattern:
        df[df['status'] == 'CONCLUÍDA']
          .groupby(['ano', 'trimestre'])['valor_total'].sum()
          .unstack()
          .loc[[year1, year2]]
          .T
          .assign(column_name=lambda x: ((x[year2] - x[year1]) / x[year1]) * 100)
    
    10. NEVER use print() in the final expression - it must be evaluable with eval()
    
    OUTPUT FORMAT:
    1. Single Python expression callable with eval()
    2. PRINT ONLY THE EXPRESSION
    """
    
    def __init__(self, df=None, db_path=CHROMADB_PATH):
        """Inicializa o analista de vendas."""
        print("\n" + "="*60)
        print("🤖 INICIALIZANDO SALES DATA ANALYST COM MEMÓRIA")
        print("="*60)
        
        # DataFrame
        self.df = df
        
        # Configurações LLM
        Settings.llm = OpenAI(
            model="gpt-4o", 
            temperature=0, 
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Configurações RAG
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection("sales_knowledge")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, embed_model=Settings.embed_model
            )
            print("✅ Índice RAG carregado com sucesso")
        except Exception as e:
            print(f"⚠️ Índice vazio: {e}")
            self.index = None
        
        # Configura LangChain com memória - NOVO
        self.llm_langchain = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Sistema de memória de conversação - NOVO
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,  # Limita para não explodir contexto
            human_prefix="Usuário",
            ai_prefix="Assistente"
        )
        
        # Histórico simplificado para contexto
        self.conversation_history = []
        
        print("✅ Analista de Vendas inicializado com memória!")
        print("="*60)
    
    def construir_indice_rag(self, word_path, business_rules):
        """Constrói índice RAG a partir de documentos."""
        print("Construindo índice RAG...")
        documents = []
        
        # Adiciona regras de negócio
        for key, value in business_rules.items():
            text = f"{key}: {value}"
            doc = Document(text=text, metadata={"tipo": "regra_negocio"})
            documents.append(doc)
        
        # Cria índice
        self.index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
        print(f"✅ RAG construído com {len(documents)} documentos")
    
    def _obter_contexto_rag(self, query_str):
        """Recupera contexto relevante do RAG."""
        if not self.index:
            return "Nenhum contexto adicional disponível."
        
        try:
            retriever = self.index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query_str)
            contexto = "\n".join([f"- {node.get_text()}" for node in nodes])
            return contexto
        except:
            return "Contexto não disponível."
    
    def _extrair_entidades(self, user_question):
        """Versão SIMPLIFICADA para fazer funcionar."""
        print(f"🔍 Extraindo entidades (simplificado): {user_question}")
        
        # Extração básica por regex
        import re
        entities = []
        
        # Anos
        anos = re.findall(r'\b(20\d{2})\b', user_question)
        entities.extend(anos)
        
        print(f"✅ Entidades detectadas: {entities}")
        return entities, {}  # chart_params vazio
        
    def _extrair_entidades_fallback(self, pergunta, result_str=""):
        """Fallback quando o JSON do LLM é inválido ou quando ocorre erro."""
        
        entities = []
        chart_params = {}  # dicionário vazio por padrão
        
        print(f"🔄 Ativando fallback para: '{pergunta}'")
        print(f"🔄 Result string recebida: {result_str[:100] if result_str else 'N/A'}")
        
        # Tenta detectar entidades básicas
        pergunta_upper = pergunta.upper()
        
        # 1. Entidades temporais
        import re
        anos = re.findall(r'\b(20\d{2})\b', pergunta)  # Regex mais precisa
        if anos:
            entities.extend(anos)
            print(f"   🕒 Anos detectados: {anos}")
            
            # Em vez disso, deixe o LLM principal decidir
            print(f"   📊 Ano {anos[0]} detectado, será processado pelo LLM principal")
        
        # 2. Meses
        meses_pt = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO', 
                    'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO']
        for mes in meses_pt:
            if mes in pergunta_upper:
                entities.append(mes)
        
        # 3. Categorias
        categorias = ['ELETRÔNICOS', 'MODA', 'CASA', 'ESPORTES', 'BELEZA']
        for cat in categorias:
            if cat in pergunta_upper:
                entities.append(cat)
                print(f"   📦 Categoria detectada: {cat}")
        
        # 4. Regiões
        regioes = ['SUDESTE', 'SUL', 'NORDESTE', 'CENTRO-OESTE', 'NORTE']
        for reg in regioes:
            if reg in pergunta_upper:
                entities.append(reg)
                print(f"   🗺️ Região detectada: {reg}")
        
        # 5. Estados (siglas)
        estados_siglas = ['SP', 'RJ', 'MG', 'PR', 'SC', 'RS', 'BA', 'PE', 'CE']
        for estado in estados_siglas:
            if f' {estado} ' in f' {pergunta_upper} ':
                entities.append(estado)
        
        # 6. Métricas
        metricas = ['VENDA', 'VALOR', 'LUCRO', 'CUSTO', 'MARGEM', 'TICKET', 'QUANTIDADE', 'CRESCIMENTO']
        for metrica in metricas:
            if metrica in pergunta_upper:
                entities.append(metrica)
        
        # Remove duplicados e garante que são strings
        entities = list(dict.fromkeys([str(e) for e in entities]))
        
        # 7. Detecta se é gráfico por keywords - APENAS SE EXPLICITAMENTE SOLICITADO
        pergunta_lower = pergunta.lower()
        
        # Lista de palavras que indicam claramente pedido de visualização
        chart_keywords_explicit = [
            'gráfico', 'grafico', 'chart', 'barras', 'colunas', 'linha', 
            'pizza', 'torta', 'dispersão', 'dispersao', 'visualização', 
            'visualizacao', 'plot', 'plotar', 'figura', 'imagem'
        ]
        
        # Palavras que podem aparecer em contextos não-gráficos
        chart_context_words = ['número', 'quantidade', 'total', 'por', 'porcentagem', 'taxa', 'índice']
        
        # Verifica se há palavras explícitas de gráfico
        has_explicit_chart = any(word in pergunta_lower for word in chart_keywords_explicit)
        
        # Verifica se tem contexto que sugere gráfico
        has_chart_context = any(word in pergunta_lower for word in chart_context_words)
        
        # Só marca como gráfico se tiver palavras explícitas
        is_chart_request = has_explicit_chart
        
        if is_chart_request:
            print(f"   📊 Pedido de gráfico detectado via fallback")
            
            # Detecta tipo de gráfico
            chart_type = "bar"  # padrão
            
            # Mapeia nomes de colunas corretos
            column_mapping = {
                'mês': 'mes',
                'mes': 'mes',
                'categorias': 'categoria',
                'categoria': 'categoria',
                'região': 'regiao',
                'regiao': 'regiao',
                'produtos': 'produto',
                'produto': 'produto',
                'vendedores': 'vendedor',
                'vendedor': 'vendedor',
                'anos': 'ano',
                'ano': 'ano',
                'trimestres': 'trimestre',
                'trimestre': 'trimestre'
            }
            
            # Detecta tipo de gráfico
            if 'linha' in pergunta_lower or 'tendência' in pergunta_lower or 'tendencia' in pergunta_lower:
                chart_type = "line"
                print(f"     ↳ Tipo: linha")
            elif 'pizza' in pergunta_lower or 'torta' in pergunta_lower:
                chart_type = "pie"
                print(f"     ↳ Tipo: pizza")
            elif 'dispersão' in pergunta_lower or 'dispersao' in pergunta_lower or 'correlação' in pergunta_lower or 'correlacao' in pergunta_lower:
                chart_type = "scatter"
                print(f"     ↳ Tipo: dispersão")
            else:
                print(f"     ↳ Tipo: barras (padrão)")
            
            # Detecta eixo X - com mapeamento correto
            x_axis = "mes"  # padrão
            
            # Procura palavras na pergunta e mapeia para nomes de colunas corretos
            for word, column in column_mapping.items():
                if word in pergunta_lower:
                    x_axis = column
                    print(f"     ↳ Eixo X detectado: '{word}' -> '{column}'")
                    break
            
            # Detecta eixo Y - com mapeamento correto
            y_axis = "valor_total"  # padrão
            
            if 'quantidade' in pergunta_lower or 'numero' in pergunta_lower or 'número' in pergunta_lower:
                y_axis = "quantidade"
                print(f"     ↳ Eixo Y: quantidade")
            elif 'lucro' in pergunta_lower:
                y_axis = "lucro"
                print(f"     ↳ Eixo Y: lucro")
            elif 'margem' in pergunta_lower:
                y_axis = "margem_lucro"
                print(f"     ↳ Eixo Y: margem_lucro")
            elif 'ticket' in pergunta_lower:
                y_axis = "valor_total"
                print(f"     ↳ Eixo Y: valor_total (para ticket médio)")
            elif 'venda' in pergunta_lower or 'vendas' in pergunta_lower:
                # Para "número de vendas", precisamos contar
                y_axis = "COUNT"
                print(f"     ↳ Eixo Y: contagem de vendas")
            else:
                print(f"     ↳ Eixo Y: valor_total (padrão)")
            
            # Cria título automático
            tipo_nome = {
                "bar": "Gráfico de Barras",
                "line": "Gráfico de Linha",
                "pie": "Gráfico de Pizza",
                "scatter": "Gráfico de Dispersão"
            }.get(chart_type, "Gráfico")
            
            title = f"{tipo_nome}: {y_axis if y_axis != 'COUNT' else 'Número de Vendas'} por {x_axis}"
            
            # Adiciona filtros se detectou
            filters = {}
            if anos:
                try:
                    filters["ano"] = int(anos[0])
                    title += f" ({anos[0]})"
                except ValueError:
                    pass
            
            # Constrói chart_params
            chart_params = {
                "chart_type": chart_type,
                "x_axis": x_axis,
                "y_axis": y_axis if y_axis != "COUNT" else "quantidade",
                "aggregation": "count" if y_axis == "COUNT" else "sum",
                "title": title,
                "filters": filters if filters else {}
            }
        
        print(f"🔄 Fallback concluído. Entidades: {entities}, Chart params: {chart_params}")
        return entities, chart_params
        
    def _detectar_grafico_por_keywords(self, pergunta):
        """Detecta parâmetros de gráfico por keywords quando o LLM falha."""
        pergunta_lower = pergunta.lower()
        
        # Mapeamento de keywords para tipos de gráfico
        chart_keywords = {
            "bar": ["barra", "coluna", "barras", "colunas", "comparar", "ranking"],
            "line": ["linha", "tendência", "evolução", "ao longo do tempo", "histórico"],
            "pie": ["pizza", "torta", "proporção", "percentual", "participação", "distribuição"],
            "scatter": ["dispersão", "correlação", "relação entre", "scatter", "ponto"]
        }
        
        # Detecta tipo
        chart_type = None
        for tipo, keywords in chart_keywords.items():
            if any(keyword in pergunta_lower for keyword in keywords):
                chart_type = tipo
                break
        
        if not chart_type:
            return {}  # Não é gráfico
        
        # Detecta eixos por keywords
        x_axis = "mes"  # padrão
        if "categoria" in pergunta_lower:
            x_axis = "categoria"
        elif "região" in pergunta_lower or "regiao" in pergunta_lower:
            x_axis = "regiao"
        elif "produto" in pergunta_lower:
            x_axis = "produto"
        elif "vendedor" in pergunta_lower:
            x_axis = "vendedor"
        
        y_axis = "valor_total"  # padrão
        if "quantidade" in pergunta_lower:
            y_axis = "quantidade"
        elif "lucro" in pergunta_lower:
            y_axis = "lucro"
        elif "margem" in pergunta_lower:
            y_axis = "margem_lucro"
        
        # Detecta filtros
        filters = {}
        import re
        ano_match = re.search(r'20\d{2}', pergunta)
        if ano_match:
            filters["ano"] = int(ano_match.group())
        
        # Cria parâmetros
        chart_params = {
            "chart_type": chart_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "title": f"Grafíco de {'Barra' if chart_type == 'bar' else 'Linha' if chart_type == 'line' else 'Pizza' if chart_type == 'pie' else 'Dispersão'}",
            "filters": filters
        }
        
        return chart_params
    
    def _normalizar_pergunta(self, question):
        """Normaliza termos na pergunta."""
        entities, _ = self._extrair_entidades(question) 
            
        if not entities:
            return question.upper()
        
        # Mapeamentos simples
        mapeamentos = {
            'CRESCIMENTO': 'CRESCIMENTO ANUAL',
            'TICKET': 'TICKET MÉDIO',
            'MARGEM': 'MARGEM DE LUCRO',
            'LUCRO': 'MARGEM DE LUCRO',
            'SP': 'SUDESTE',
            'RJ': 'SUDESTE',
            'MG': 'SUDESTE'
        }
        
        question_norm = question.upper()
        for entity in entities:
            # Verifica se a entidade é hashable (string)
            if isinstance(entity, str):
                if entity in mapeamentos:
                    question_norm = question_norm.replace(entity, mapeamentos[entity])
        
        return question_norm

    def _formatar_historico_conversa(self, max_messages=4):
        """Formata histórico mantendo apenas perguntas."""
        if not self.conversation_history:
            return "Nenhuma conversa anterior."
        
        # Filtra apenas perguntas (índices pares)
        perguntas = [self.conversation_history[i] for i in range(0, len(self.conversation_history), 2)]
        
        # Pega as últimas N perguntas
        recentes = perguntas[-max_messages:] if len(perguntas) > max_messages else perguntas
        
        formatted = []
        for i, pergunta in enumerate(recentes):
            formatted.append(f"Usuário {i+1}: {pergunta}")
        
        return " | ".join(formatted)  # Formato mais compacto
    
    def consultar(self, question, verbose=True):
        """Processa consulta do usuário com memória de contexto."""
        # Salva pergunta no histórico
        self.conversation_history.append(question)
        self.conversation_memory.chat_memory.add_message(
            HumanMessage(content=question)
        )

        # EXTRAI ENTIDADES E PARÂMETROS DE GRÁFICO
        entities, chart_params = self._extrair_entidades(question)

        # Obtém histórico formatado
        historico_formatado = self._formatar_historico_conversa()

        if verbose:
            print(f"📝 Pergunta original: {question}")
            print(f"📚 Histórico ({len(self.conversation_history)} mensagens):")
            print(f"   {historico_formatado[:200]}...")
            print(f"📊 Parâmetros gráfico: {chart_params}")

        # Normaliza pergunta
        pergunta_ajustada = self._normalizar_pergunta(question)

        # Obtém contexto RAG
        contexto_recuperado = self._obter_contexto_rag(pergunta_ajustada)

        if verbose:
            print("🔍 Contexto RAG recuperado")

        # Monta prompt
        instrucoes_com_contexto = self.CONTEXT_PANDAS_INSTRUCOES.format(
            contexto_rag=contexto_recuperado,
            historico_conversa=historico_formatado,
            user_query=pergunta_ajustada,
        )

        # Se for gráfico, adiciona instrução especial
        if chart_params and chart_params.get("chart_type"):
            instrucoes_com_contexto += (
                f"\n\nNOTE: User requested a {chart_params['chart_type']} chart. "
                "Focus on data aggregation for visualization."
            )
        else:
            instrucoes_com_contexto += (
                "\n\nIMPORTANT: Do NOT generate plot() or chart code unless "
                "explicitly requested with words like 'gráfico', 'chart', 'visualização'."
            )

        # Template mínimo
        response_template = "Responda diretamente com os resultados da análise."
        response_prompt = PromptTemplate(response_template)

        # Cria query engine
        query_engine = PandasQueryEngine(
            df=self.df,
            instruction_str=instrucoes_com_contexto,
            response_synthesis_prompt=response_prompt,
            synthesize_response=False,
            verbose=verbose,
        )

        try:
            response = query_engine.query(pergunta_ajustada)
            pandas_code_str = response.metadata.get("pandas_instruction_str", "")
            
            # Remove avisos de depreciação do código
            if pandas_code_str and 'applymap' in pandas_code_str:
                print(f"🔄 Corrigindo applymap depreciado...")
                # Substitui applymap por map
                pandas_code_str = pandas_code_str.replace('.applymap(', '.map(')
                
                # Para DataFrames, se estiver usando lambda para formatação
                if 'lambda x:' in pandas_code_str and 'f"' in pandas_code_str:
                    # Melhor usar apply com axis=1
                    pandas_code_str = pandas_code_str.replace('.map(', '.apply(')
            
            # RESPOSTA SIMPLIFICADA - NÃO FAÇA FORMATAÇÃO AQUI!
            # A formatação será feita em app.py pela função formatar_resultado_pandas
            resposta_final = f"📊 **Consulta processada: {question}**"
            
            # Log para debug
            print(f"✅ Código pandas gerado: {pandas_code_str[:200]}...")
            
            # Adiciona uma resposta resumida ao histórico
            resposta_resumida = f"Resposta para: {question[:50]}..."
            
            self.conversation_history.append(resposta_resumida)
            self.conversation_memory.chat_memory.add_message(
                AIMessage(content=resposta_resumida)
            )

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            if verbose:
                print(f"💾 Histórico atualizado: {len(self.conversation_history)} mensagens")

            # RETORNA APENAS O CÓDIGO, A FORMATAÇÃO SERÁ FEITA EM app.py
            return resposta_final, pandas_code_str

        except Exception as e:
            print(f"⚠️ Erro na consulta: {e}")
            import traceback
            traceback.print_exc()

            error_msg = f"Desculpe, houve um erro ao processar sua consulta: {str(e)}"
            self.conversation_history.append(error_msg)
            self.conversation_memory.chat_memory.add_message(
                AIMessage(content=error_msg)
            )

            return error_msg, ""

# Exemplo de uso
if __name__ == "__main__":
    # Teste rápido
    from data_loader import generate_sample_data
    
    print("🧪 Testando SalesDataAnalyst...")
    df = generate_sample_data(100)
    analista = SalesDataAnalyst(df)
    
    # Testa algumas consultas
    test_queries = [
        "Vendas totais por mês",
        "Ticket médio por região",
        "Margem de lucro por categoria"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Consulta: {query}")
        resposta, codigo = analista.consultar(query, verbose=False)
        print(f"Resposta: {resposta[:100]}...")
        print(f"Código: {codigo[:100]}...")