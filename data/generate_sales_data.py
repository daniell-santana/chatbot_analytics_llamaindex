# generate_sales_data.py - Gerador de dados de vendas SEM mimesis
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

def generate_sales_dataset(n_records=5000, start_date='2020-01-01', end_date='2025-12-31'):
    """Gera um dataset realista de vendas sem dependência do mimesis."""
    
    # Categorias e produtos
    categories = {
        'ELETRÔNICOS': ['SMARTPHONE', 'NOTEBOOK', 'TABLET', 'SMART TV', 'FONES DE OUVIDO'],
        'MODA': ['CAMISETA', 'CALÇA JEANS', 'VESTIDO', 'TÊNIS', 'BOLSA'],
        'CASA': ['SOFÁ', 'MESA', 'CADEIRA', 'CAMA', 'GELADEIRA'],
        'ESPORTES': ['BICICLETA', 'ESTEIRA', 'HALTERES', 'BOLA', 'RAQUETE'],
        'BELEZA': ['PERFUME', 'CREME', 'MAQUIAGEM', 'SHAMPOO', 'BATOM']
    }
    
    # Regiões e estados do Brasil
    regions = {
        'SUDESTE': ['SP', 'RJ', 'MG', 'ES'],
        'SUL': ['PR', 'SC', 'RS'],
        'NORDESTE': ['BA', 'PE', 'CE', 'MA', 'PB', 'RN', 'AL', 'SE', 'PI'],
        'CENTRO-OESTE': ['GO', 'MT', 'MS', 'DF'],
        'NORTE': ['AM', 'PA', 'RO', 'AC', 'RR', 'AP', 'TO']
    }
    
    # Nomes para vendedores e clientes
    first_names = ['ANA', 'CARLOS', 'MARIA', 'JOSÉ', 'PATRÍCIA', 'PAULO', 'LUCAS', 'JULIANA', 
                   'FERNANDO', 'AMANDA', 'ROBERTO', 'CAMILA', 'RICARDO', 'BEATRIZ', 'GUSTAVO']
    last_names = ['SILVA', 'SANTOS', 'OLIVEIRA', 'SOUZA', 'RODRIGUES', 'FERNANDES', 'PEREIRA',
                  'ALMEIDA', 'LIMA', 'COSTA', 'GOMES', 'MARTINS', 'RIBEIRO', 'ALVES']
    
    cities = {
        'SP': ['SÃO PAULO', 'CAMPINAS', 'SANTOS', 'SOROCABA'],
        'RJ': ['RIO DE JANEIRO', 'NITERÓI', 'PETRÓPOLIS'],
        'MG': ['BELO HORIZONTE', 'UBERLÂNDIA', 'CONTAGEM'],
        'PR': ['CURITIBA', 'LONDRINA', 'MARINGÁ'],
        'SC': ['FLORIANÓPOLIS', 'JOINVILLE', 'BLUMENAU'],
        'RS': ['PORTO ALEGRE', 'CACHOEIRINHA', 'CANOAS']
    }
    
    # Prepara datas
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days_diff = (end - start).days
    
    data = []
    
    print(f"🔧 Gerando {n_records} registros de vendas...")
    
    for i in range(n_records):
        # Data aleatória no período
        random_days = random.randint(0, days_diff)
        sale_date = start + timedelta(days=random_days)
        
        # Seleciona categoria e produto
        category = random.choice(list(categories.keys()))
        product = random.choice(categories[category])
        
        # Preços base por categoria
        price_ranges = {
            'ELETRÔNICOS': (800, 5000),
            'MODA': (50, 300),
            'CASA': (200, 3000),
            'ESPORTES': (100, 2000),
            'BELEZA': (30, 200)
        }
        
        min_price, max_price = price_ranges[category]
        unit_price = round(random.uniform(min_price, max_price), 2)
        quantity = random.randint(1, 5)
        
        # Descontos sazonais (mais descontos em nov/dez)
        discount = 0
        month = sale_date.month
        if month in [11, 12]:  # Black Friday/Natal
            discount = random.uniform(0.15, 0.35)
        elif month in [1, 7]:  # Janeiro/Julho
            discount = random.uniform(0.05, 0.20)
        elif random.random() < 0.1:  # 10% chance de desconto aleatório
            discount = random.uniform(0.05, 0.15)
        
        total_sale = unit_price * quantity * (1 - discount)
        
        # Seleciona região e estado
        region = random.choice(list(regions.keys()))
        estado = random.choice(regions[region])
        cidade = random.choice(cities.get(estado, ['CIDADE DESCONHECIDA']))
        
        # Gera nomes
        vendedor = f"{random.choice(first_names)} {random.choice(last_names)}"
        cliente = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Forma de pagamento
        payment_methods = ['CARTAO CREDITO', 'CARTAO DEBITO', 'PIX', 'BOLETO', 'DINHEIRO']
        forma_pagamento = random.choice(payment_methods)
        
        # Status (95% concluída, 5% cancelada)
        status = 'CONCLUÍDA' if random.random() < 0.95 else 'CANCELADA'
        
        data.append({
            'id_venda': f"V{2020 + (i % 4)}{i:05d}",  # ID com ano
            'data': sale_date.strftime('%Y-%m-%d'),
            'ano': sale_date.year,
            'mes': sale_date.month,
            'trimestre': (sale_date.month - 1) // 3 + 1,
            'dia_semana': sale_date.strftime('%A').upper(),
            'categoria': category,
            'produto': product,
            'regiao': region,
            'estado': estado,
            'cidade': cidade,
            'vendedor': vendedor,
            'cliente': cliente,
            'quantidade': quantity,
            'preco_unitario': unit_price,
            'desconto_percentual': round(discount * 100, 2),
            'valor_total': round(total_sale, 2),
            'forma_pagamento': forma_pagamento,
            'custo_unitario': round(unit_price * random.uniform(0.4, 0.7), 2),
            'status': status
        })
        
        # Progresso
        if (i + 1) % 1000 == 0:
            print(f"  ✅ Gerados {i + 1} registros...")
    
    df = pd.DataFrame(data)
    
    # Adiciona cálculos financeiros
    df['custo_total'] = df['custo_unitario'] * df['quantidade']
    df['lucro'] = df['valor_total'] - df['custo_total']
    df['margem_lucro'] = (df['lucro'] / df['valor_total'] * 100).round(2)
    
    # Adiciona trimestre se não calculou corretamente
    df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype(int)
    
    return df

def main():
    """Função principal para gerar e salvar dados."""
    print("="*60)
    print("📊 GERADOR DE DADOS DE VENDAS - SalesInsight AI")
    print("="*60)
    
    try:
        # Gera dataset
        df = generate_sales_dataset(10000)  # 10k registros
        
        # Salva como CSV
        output_path = 'data/sales_data.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Estatísticas
        print(f"\n✅ Dataset gerado com sucesso!")
        print(f"📁 Salvo em: {output_path}")
        print(f"📈 Estatísticas:")
        print(f"   • Registros: {len(df):,}")
        print(f"   • Período: {df['data'].min()} até {df['data'].max()}")
        print(f"   • Vendas totais: R$ {df['valor_total'].sum():,.2f}")
        print(f"   • Ticket médio: R$ {df['valor_total'].mean():,.2f}")
        print(f"   • Categorias: {df['categoria'].nunique()}")
        print(f"   • Regiões: {df['regiao'].nunique()}")
        print(f"   • Status: {df['status'].value_counts().to_dict()}")
        
        # Amostra dos dados
        print(f"\n🔍 Amostra dos dados (primeiras 3 linhas):")
        print(df[['data', 'categoria', 'produto', 'valor_total', 'status']].head(3).to_string(index=False))
        
        # Salva também um arquivo de metadados
        metadata = {
            'total_records': len(df),
            'date_range': f"{df['data'].min()} to {df['data'].max()}",
            'total_sales': float(df['valor_total'].sum()),
            'categories': df['categoria'].unique().tolist(),
            'regions': df['regiao'].unique().tolist(),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('data/dataset_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Metadados salvos em: data/dataset_metadata.json")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar dados: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()