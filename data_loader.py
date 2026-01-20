# data_loader.py - Carregamento e processamento de dados
import pandas as pd
import numpy as np
from typing import Optional
import logging
import os

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
    
    def _load_data(self):
        """Carrega os dados do arquivo."""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")
            
            # Detecta extensão do arquivo
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path, encoding='utf-8')
            elif self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.json'):
                self.df = pd.read_json(self.file_path)
            else:
                raise ValueError("Formato de arquivo não suportado. Use CSV, Excel ou JSON.")
            
            logger.info(f"Dados carregados: {self.df.shape}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def _clean_data(self):
        """Limpeza e preparação dos dados."""
        df = self.df
        
        # Converte datas
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df['ano'] = df['data'].dt.year.fillna(0).astype(int)
            df['mes'] = df['data'].dt.month.fillna(0).astype(int)
            df['dia'] = df['data'].dt.day.fillna(0).astype(int)
            df['dia_semana'] = df['data'].dt.day_name()
            df['trimestre'] = df['data'].dt.quarter
        
        # Converte colunas numéricas
        numeric_cols = [
            'quantidade', 'preco_unitario', 'desconto_percentual',
            'valor_total', 'custo_unitario', 'custo_total', 'lucro', 'margem_lucro'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Garante strings para colunas categóricas
        categoric_cols = [
            'categoria', 'produto', 'regiao', 'estado', 'cidade',
            'vendedor', 'cliente', 'forma_pagamento', 'status'
        ]
        
        for col in categoric_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Desconhecido').astype(str).str.upper()
        
        # Calcula trimestre se não existir
        if 'trimestre' not in df.columns and 'mes' in df.columns:
            df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype(int)
        
        self.df = df
    
    def _add_derived_features(self):
        """Adiciona features derivadas para análise."""
        df = self.df
        
        # Flag para vendas com desconto
        if 'desconto_percentual' in df.columns:
            df['tem_desconto'] = df['desconto_percentual'] > 0
        
        # Categoria de valor da venda
        if 'valor_total' in df.columns:
            conditions = [
                df['valor_total'] < 100,
                (df['valor_total'] >= 100) & (df['valor_total'] < 500),
                (df['valor_total'] >= 500) & (df['valor_total'] < 1000),
                df['valor_total'] >= 1000
            ]
            choices = ['Baixo', 'Médio', 'Alto', 'Muito Alto']
            df['categoria_valor'] = np.select(conditions, choices, default='Desconhecido')
        
        # Temporada (baseada no mês)
        if 'mes' in df.columns:
            conditions = [
                df['mes'].isin([11, 12]),  # Fim de ano
                df['mes'].isin([6, 7]),    # Meio do ano
                df['mes'].isin([1, 2]),    # Começo do ano
                True
            ]
            choices = ['Alta Temporada', 'Média Temporada', 'Baixa Temporada', 'Normal']
            df['temporada'] = np.select(conditions, choices, default='Normal')
        
        self.df = df
    
    def processed_data(self) -> pd.DataFrame:
        """Pipeline completo de processamento."""
        self._load_data()
        self._clean_data()
        self._add_derived_features()
        
        # Informações de diagnóstico
        print("\n" + "="*60)
        print("📊 RESUMO DO DATASET PROCESSADO")
        print("="*60)
        print(f"• Período: {self.df['ano'].min()} - {self.df['ano'].max()}")
        print(f"• Total vendas: R$ {self.df['valor_total'].sum():,.2f}")
        print(f"• Registros: {len(self.df):,}")
        print(f"• Categorias: {self.df['categoria'].nunique()}")
        print(f"• Status: {self.df['status'].value_counts().to_dict()}")
        print("="*60)
        
        return self.df

# Função de exemplo para gerar dados
def generate_sample_data(n_records=1000):
    """Gera dados de exemplo se não existir arquivo."""
    import random
    from datetime import datetime, timedelta
    
    categories = ['ELETRÔNICOS', 'MODA', 'CASA', 'ESPORTES', 'BELEZA']
    products = {
        'ELETRÔNICOS': ['SMARTPHONE', 'NOTEBOOK', 'TABLET', 'TV', 'FONES'],
        'MODA': ['CAMISETA', 'CALÇA', 'VESTIDO', 'TÊNIS', 'BOLSA'],
        'CASA': ['SOFÁ', 'MESA', 'CADEIRA', 'CAMA', 'GELADEIRA'],
        'ESPORTES': ['BICICLETA', 'ESTEIRA', 'HALTERES', 'BOLA', 'RAQUETE'],
        'BELEZA': ['PERFUME', 'CREME', 'MAQUIAGEM', 'SHAMPOO', 'BATOM']
    }
    regions = ['SUDESTE', 'SUL', 'NORDESTE', 'CENTRO-OESTE', 'NORTE']
    
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(n_records):
        sale_date = start_date + timedelta(days=random.randint(0, 1460))
        category = random.choice(categories)
        product = random.choice(products[category])
        
        price_range = {
            'ELETRÔNICOS': (800, 5000),
            'MODA': (50, 300),
            'CASA': (200, 3000),
            'ESPORTES': (100, 2000),
            'BELEZA': (30, 200)
        }
        
        min_price, max_price = price_range[category]
        unit_price = round(random.uniform(min_price, max_price), 2)
        quantity = random.randint(1, 5)
        discount = random.choice([0, 0, 0, 0.1, 0.15, 0.2])  # 25% chance de desconto
        
        total = unit_price * quantity * (1 - discount)
        
        data.append({
            'id_venda': f"V{i:06d}",
            'data': sale_date.strftime('%Y-%m-%d'),
            'categoria': category,
            'produto': product,
            'regiao': random.choice(regions),
            'estado': random.choice(['SP', 'RJ', 'MG', 'PR', 'SC', 'RS']),
            'vendedor': f"Vendedor_{random.randint(1, 20)}",
            'quantidade': quantity,
            'preco_unitario': unit_price,
            'desconto_percentual': discount * 100,
            'valor_total': round(total, 2),
            'forma_pagamento': random.choice(['CARTAO CREDITO', 'CARTAO DEBITO', 'PIX', 'BOLETO']),
            'custo_unitario': round(unit_price * random.uniform(0.4, 0.7), 2),
            'status': random.choice(['CONCLUÍDA', 'CONCLUÍDA', 'CONCLUÍDA', 'CANCELADA'])
        })
    
    df = pd.DataFrame(data)
    df['custo_total'] = df['custo_unitario'] * df['quantidade']
    df['lucro'] = df['valor_total'] - df['custo_total']
    df['margem_lucro'] = (df['lucro'] / df['valor_total'] * 100).round(2)
    
    return df