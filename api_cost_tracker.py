# api_cost_tracker.py
"""
Monitor de Custos da API OpenAI para o Chatbot Analítico
--------------------------------------------------------
Este script rastreia os custos das chamadas à API OpenAI.
Adicione 'from api_cost_tracker import track_cost' no início dos seus arquivos.

Baseado nos preços oficiais (outubro 2023):
- GPT-4o: Input $2.50 / 1M tokens, Output $10.00 / 1M tokens
- GPT-4-turbo: Input $10.00 / 1M tokens, Output $30.00 / 1M tokens
"""

import json
import os
import time
from datetime import datetime, date
from typing import Dict, Any, Optional
import pandas as pd

class OpenAICostTracker:
    """Classe para rastrear custos da API OpenAI."""
    
    # Preços por 1 milhão de tokens (USD)
    PRICING = {
        "gpt-4o": {
            "input": 2.50,      # $2.50 por 1M tokens de entrada
            "output": 10.00,    # $10.00 por 1M tokens de saída
        },
        "gpt-4o-mini": {
            "input": 0.15,      # $0.15 por 1M tokens
            "output": 0.60,     # $0.60 por 1M tokens
        },
        "gpt-4-turbo": {
            "input": 10.00,     # $10.00 por 1M tokens
            "output": 30.00,    # $30.00 por 1M tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.50,      # $0.50 por 1M tokens
            "output": 1.50,     # $1.50 por 1M tokens
        }
    }
    
    def __init__(self, log_file: str = "data/api_costs.json"):
        """
        Inicializa o monitor de custos.
        
        Args:
            log_file: Caminho do arquivo para salvar os logs
        """
        # Cria pasta data/ se não existir
        os.makedirs("data", exist_ok=True) 

        self.log_file = log_file
        self.daily_stats = self._load_daily_stats()
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        
        print("💰 Monitor de Custos OpenAI inicializado")
        print(f"📊 Arquivo de log: {log_file}")
    
    def _load_daily_stats(self) -> Dict[str, Any]:
        """Carrega estatísticas diárias do arquivo JSON."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._create_empty_stats()
        return self._create_empty_stats()
    
    def _create_empty_stats(self) -> Dict[str, Any]:
        """Cria estrutura vazia para estatísticas."""
        today = date.today().isoformat()
        return {
            "daily_costs": {today: 0.0},
            "daily_tokens": {today: {"input": 0, "output": 0}},
            "model_usage": {},
            "total_cost": 0.0,
            "total_tokens": {"input": 0, "output": 0},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_stats(self):
        """Salva estatísticas no arquivo JSON."""
        try:
            # Garante que a pasta data existe
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.daily_stats, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ Erro ao salvar estatísticas: {e}")
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calcula o custo de uma chamada à API.
        
        Args:
            model: Nome do modelo (ex: "gpt-4o")
            prompt_tokens: Tokens de entrada (prompt)
            completion_tokens: Tokens de saída (completion)
            
        Returns:
            Custo em USD
        """
        # Usa gpt-4o como padrão se modelo não estiver na lista
        model_key = model if model in self.PRICING else "gpt-4o"
        
        input_cost = (prompt_tokens / 1_000_000) * self.PRICING[model_key]["input"]
        output_cost = (completion_tokens / 1_000_000) * self.PRICING[model_key]["output"]
        
        return round(input_cost + output_cost, 6)
    
    def track_call(self, model: str, prompt_tokens: int, completion_tokens: int, 
                   metadata: Optional[Dict] = None):
        """
        Registra uma chamada à API e calcula o custo.
        
        Args:
            model: Nome do modelo
            prompt_tokens: Tokens de entrada
            completion_tokens: Tokens de saída
            metadata: Metadados adicionais (opcional)
        """
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        today = date.today().isoformat()
        
        # Atualiza estatísticas diárias
        self.daily_stats["daily_costs"][today] = self.daily_stats["daily_costs"].get(today, 0.0) + cost
        self.daily_stats["total_cost"] = self.daily_stats.get("total_cost", 0.0) + cost
        
        # Atualiza tokens diários
        daily_tokens = self.daily_stats["daily_tokens"].get(today, {"input": 0, "output": 0})
        daily_tokens["input"] = daily_tokens.get("input", 0) + prompt_tokens
        daily_tokens["output"] = daily_tokens.get("output", 0) + completion_tokens
        self.daily_stats["daily_tokens"][today] = daily_tokens
        
        # Atualiza tokens totais
        self.daily_stats["total_tokens"]["input"] = self.daily_stats["total_tokens"].get("input", 0) + prompt_tokens
        self.daily_stats["total_tokens"]["output"] = self.daily_stats["total_tokens"].get("output", 0) + completion_tokens
        
        # Atualiza uso por modelo
        model_stats = self.daily_stats["model_usage"].get(model, {
            "calls": 0,
            "total_cost": 0.0,
            "total_tokens": {"input": 0, "output": 0}
        })
        model_stats["calls"] += 1
        model_stats["total_cost"] = round(model_stats.get("total_cost", 0.0) + cost, 6)
        model_stats["total_tokens"]["input"] = model_stats["total_tokens"].get("input", 0) + prompt_tokens
        model_stats["total_tokens"]["output"] = model_stats["total_tokens"].get("output", 0) + completion_tokens
        self.daily_stats["model_usage"][model] = model_stats
        
        # Adiciona metadados se fornecidos
        if metadata:
            if "detailed_calls" not in self.daily_stats:
                self.daily_stats["detailed_calls"] = []
            
            call_info = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "metadata": metadata
            }
            self.daily_stats["detailed_calls"].append(call_info)
        
        # Atualiza data da última modificação
        self.daily_stats["last_updated"] = datetime.now().isoformat()
        
        # Salva estatísticas
        self._save_stats()
        
        # Log no console
        print(f"💰 API Call: {model} | Tokens: {prompt_tokens}+{completion_tokens} | Cost: ${cost:.6f}")
        
        return cost
    
    def get_daily_summary(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna resumo dos custos de um dia específico.
        
        Args:
            date_str: Data no formato YYYY-MM-DD (padrão: hoje)
            
        Returns:
            Dicionário com resumo do dia
        """
        if date_str is None:
            date_str = date.today().isoformat()
        
        return {
            "date": date_str,
            "cost": self.daily_stats["daily_costs"].get(date_str, 0.0),
            "tokens": self.daily_stats["daily_tokens"].get(date_str, {"input": 0, "output": 0}),
            "calls_today": sum(1 for call in self.daily_stats.get("detailed_calls", []) 
                             if call["timestamp"].startswith(date_str))
        }
    
    def get_monthly_summary(self, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """
        Retorna resumo dos custos de um mês específico.
        
        Args:
            year: Ano (padrão: ano atual)
            month: Mês (padrão: mês atual)
            
        Returns:
            Dicionário com resumo do mês
        """
        if year is None:
            year = date.today().year
        if month is None:
            month = date.today().month
        
        month_prefix = f"{year:04d}-{month:02d}"
        
        monthly_cost = 0.0
        monthly_tokens = {"input": 0, "output": 0}
        daily_calls = []
        
        for day, cost in self.daily_stats["daily_costs"].items():
            if day.startswith(month_prefix):
                monthly_cost += cost
                monthly_tokens["input"] += self.daily_stats["daily_tokens"].get(day, {}).get("input", 0)
                monthly_tokens["output"] += self.daily_stats["daily_tokens"].get(day, {}).get("output", 0)
                daily_calls.append({
                    "date": day,
                    "cost": cost,
                    "tokens": self.daily_stats["daily_tokens"].get(day, {})
                })
        
        return {
            "year": year,
            "month": month,
            "total_cost": monthly_cost,
            "total_tokens": monthly_tokens,
            "days": len(daily_calls),
            "daily_breakdown": daily_calls
        }
    
    def print_summary(self):
        """Imprime resumo dos custos no console."""
        print("\n" + "="*60)
        print("💰 RESUMO DE CUSTOS - API OpenAI")
        print("="*60)
        
        # Total geral
        print(f"\n📊 TOTAL GERAL:")
        print(f"   • Custo total: ${self.daily_stats['total_cost']:.4f} USD")
        print(f"   • Tokens entrada: {self.daily_stats['total_tokens']['input']:,}")
        print(f"   • Tokens saída: {self.daily_stats['total_tokens']['output']:,}")
        
        # Por modelo
        print(f"\n🤖 USO POR MODELO:")
        for model, stats in self.daily_stats.get("model_usage", {}).items():
            print(f"   • {model}:")
            print(f"     - Chamadas: {stats['calls']}")
            print(f"     - Custo: ${stats['total_cost']:.4f}")
            print(f"     - Tokens: {stats['total_tokens']['input']:,} + {stats['total_tokens']['output']:,}")
        
        # Últimos 7 dias
        print(f"\n📅 ÚLTIMOS 7 DIAS:")
        today = date.today()
        for i in range(7):
            day = today - pd.Timedelta(days=i)
            day_str = day.isoformat()
            cost = self.daily_stats["daily_costs"].get(day_str, 0.0)
            tokens = self.daily_stats["daily_tokens"].get(day_str, {"input": 0, "output": 0})
            
            if cost > 0 or tokens["input"] > 0:
                print(f"   • {day_str}: ${cost:.4f} | Tokens: {tokens['input']:,}+{tokens['output']:,}")
        
        print("="*60)
    
    def export_to_csv(self, output_file: str = "data/api_costs_report.csv"):
        """
        Exporta dados de custos para CSV.
        
        Args:
            output_file: Nome do arquivo CSV de saída
        """
        try:
            # Garante que a pasta data existe
            os.makedirs("data", exist_ok=True) 

            # Prepara dados para DataFrame
            data = []
            for day, cost in self.daily_stats["daily_costs"].items():
                tokens = self.daily_stats["daily_tokens"].get(day, {"input": 0, "output": 0})
                data.append({
                    "date": day,
                    "cost_usd": cost,
                    "tokens_input": tokens["input"],
                    "tokens_output": tokens["output"],
                    "total_tokens": tokens["input"] + tokens["output"]
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.sort_values("date", ascending=False)
                df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"✅ Dados exportados para: {output_file}")
                return df
            else:
                print("⚠️ Nenhum dado para exportar")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {e}")
            return None


# Instância global para uso fácil
cost_tracker = OpenAICostTracker()


# Decorador para facilitar o rastreamento
def track_openai_cost(func):
    """
    Decorador para rastrear custos de funções que usam OpenAI.
    
    Uso:
        @track_openai_cost
        def minha_funcao_com_openai():
            # sua função aqui
            pass
    """
    def wrapper(*args, **kwargs):
        # Tenta extrair informações da chamada
        model = kwargs.get('model', 'gpt-4o')
        
        # Executa a função original
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Tenta extrair tokens da resposta (se disponível)
        prompt_tokens = 0
        completion_tokens = 0
        
        if hasattr(result, 'usage'):
            # Se a resposta tem atributo usage (como do OpenAI SDK)
            prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(result.usage, 'completion_tokens', 0)
        
        # Registra a chamada
        cost_tracker.track_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "function": func.__name__,
                "execution_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return result
    
    return wrapper


# Função de inicialização rápida
def init_cost_tracking(log_file: str = "data/api_costs.json"):
    """
    Inicializa o monitor de custos.
    
    Args:
        log_file: Caminho do arquivo de log
    """
    global cost_tracker
    cost_tracker = OpenAICostTracker(log_file)
    return cost_tracker


if __name__ == "__main__":
    # Teste do monitor
    tracker = OpenAICostTracker()
    
    # Exemplo de chamadas simuladas
    print("🧪 Testando monitor de custos...")
    
    # Simula algumas chamadas
    tracker.track_call("gpt-4o", 1500, 800, {"query": "Vendas por mês"})
    tracker.track_call("gpt-4o", 1200, 600, {"query": "Ticket médio"})
    tracker.track_call("gpt-4o", 2000, 1500, {"query": "Análise trimestral"})
    
    # Mostra resumo
    tracker.print_summary()
    
    # Exporta para CSV
    tracker.export_to_csv()
    
    print("\n✅ Monitor de custos testado com sucesso!")