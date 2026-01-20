#Objetos auxiliares e mapeamentos

# Dicionário de regiões e estados
REGIOES_ESTADOS = {
    'SUDESTE': ['SP', 'RJ', 'MG', 'ES'],
    'SUL': ['PR', 'SC', 'RS'],
    'NORDESTE': ['BA', 'PE', 'CE', 'MA', 'PB', 'RN', 'AL', 'SE', 'PI'],
    'CENTRO-OESTE': ['GO', 'MT', 'MS', 'DF'],
    'NORTE': ['AM', 'PA', 'RO', 'AC', 'RR', 'AP', 'TO']
}

# Mapeamento de estados para regiões
ESTADO_PARA_REGIAO = {}
for regiao, estados in REGIOES_ESTADOS.items():
    for estado in estados:
        ESTADO_PARA_REGIAO[estado] = regiao

# Categorias e subcategorias de produtos
CATEGORIAS_PRODUTOS = {
    'ELETRÔNICOS': {
        'descricao': 'Produtos eletrônicos e tecnologia',
        'subcategorias': ['CELULARES', 'COMPUTADORES', 'TV E VIDEO', 'ÁUDIO', 'ACESSÓRIOS']
    },
    'MODA': {
        'descricao': 'Vestuário e acessórios de moda',
        'subcategorias': ['ROUPA MASCULINA', 'ROUPA FEMININA', 'CALÇADOS', 'ACESSÓRIOS', 'BIJUTERIAS']
    },
    'CASA': {
        'descricao': 'Móveis e artigos para casa',
        'subcategorias': ['MÓVEIS', 'ELETRODOMÉSTICOS', 'DECORAÇÃO', 'COZINHA', 'JARDIM']
    },
    'ESPORTES': {
        'descricao': 'Artigos esportivos e equipamentos',
        'subcategorias': ['FITNESS', 'ESPORTES COLETIVOS', 'CAMPO E PRAIA', 'CICLISMO', 'AQUÁTICOS']
    },
    'BELEZA': {
        'descricao': 'Produtos de beleza e cuidados pessoais',
        'subcategorias': ['COSMÉTICOS', 'PERFUMARIA', 'CUIDADOS CAPILARES', 'MAQUIAGEM', 'TRATAMENTOS']
    }
}

# Formas de pagamento
FORMAS_PAGAMENTO = [
    'CARTAO CREDITO',
    'CARTAO DEBITO', 
    'PIX',
    'BOLETO',
    'DINHEIRO',
    'TRANSFERENCIA'
]

# Status de vendas
STATUS_VENDAS = [
    'CONCLUÍDA',
    'CANCELADA',
    'PENDENTE',
    'ESTORNADA',
    'EM PROCESSAMENTO'
]

# Meses em português
MESES_PORTUGUES = {
    1: 'JANEIRO',
    2: 'FEVEREIRO',
    3: 'MARÇO',
    4: 'ABRIL',
    5: 'MAIO',
    6: 'JUNHO',
    7: 'JULHO',
    8: 'AGOSTO',
    9: 'SETEMBRO',
    10: 'OUTUBRO',
    11: 'NOVEMBRO',
    12: 'DEZEMBRO'
}

# Dias da semana
DIAS_SEMANA = [
    'SEGUNDA',
    'TERÇA', 
    'QUARTA',
    'QUINTA',
    'SEXTA',
    'SÁBADO',
    'DOMINGO'
]

# Temporadas (para análise sazonal)
TEMPORADAS = {
    'ALTA': [11, 12, 1],  # Nov, Dez, Jan (Natal/Ano Novo)
    'MEDIA': [6, 7, 8],   # Jun, Jul, Ago (Férias)
    'BAIXA': [2, 9],      # Fev, Set (Pós-férias)
    'NORMAL': [3, 4, 5, 10]
}

# Faixas de valor para análise
FAIXAS_VALOR = {
    'BAIXO': (0, 100),
    'MÉDIO': (100, 500),
    'ALTO': (500, 1000),
    'MUITO ALTO': (1000, float('inf'))
}

# KPIs e métricas padrão
KPIS_PADRAO = [
    'VENDAS BRUTAS',
    'VENDAS LÍQUIDAS',
    'TICKET MÉDIO',
    'MARGEM DE LUCRO',
    'CRESCIMENTO',
    'CUSTO DA MERCADORIA VENDIDA',
    'TAXA DE CONVERSÃO',
    'ROTATIVIDADE DE ESTOQUE',
    'LUCRO LÍQUIDO',
    'RETORNO SOBRE INVESTIMENTO'
]

# Funções úteis
def normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparação."""
    if not texto:
        return ""
    return texto.upper().strip()

def mapear_regiao(estado: str) -> str:
    """Mapeia estado para região."""
    estado_norm = normalizar_texto(estado)
    return ESTADO_PARA_REGIAO.get(estado_norm, 'DESCONHECIDA')

def obter_mes_portugues(mes_num: int) -> str:
    """Retorna nome do mês em português."""
    return MESES_PORTUGUES.get(mes_num, 'DESCONHECIDO')

def classificar_faixa_valor(valor: float) -> str:
    """Classifica valor em faixa."""
    for faixa, (min_val, max_val) in FAIXAS_VALOR.items():
        if min_val <= valor < max_val:
            return faixa
    return 'DESCONHECIDO'

# Listas para validação
CATEGORIAS_VALIDAS = list(CATEGORIAS_PRODUTOS.keys())
PRODUTOS_VALIDOS = []
for cat_info in CATEGORIAS_PRODUTOS.values():
    PRODUTOS_VALIDOS.extend(cat_info['subcategorias'])

REGIOES_VALIDAS = list(REGIOES_ESTADOS.keys())
ESTADOS_VALIDOS = []
for estados in REGIOES_ESTADOS.values():
    ESTADOS_VALIDOS.extend(estados)