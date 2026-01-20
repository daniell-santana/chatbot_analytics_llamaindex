# check_costs.py
"""
Script para verificar e gerenciar custos da API OpenAI.
Execute: python check_costs.py

Funcionalidades:
1. Ver resumo geral
2. Exportar para CSV
3. Ver detalhes por período
4. Configurar alertas de limite
5. Mostrar dashboard interativo
"""

import sys
import os
from datetime import datetime, date, timedelta
import pandas as pd
import json

def clear_screen():
    """Limpa a tela do console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Imprime cabeçalho formatado."""
    print("\n" + "="*60)
    print(f"📊 {title}")
    print("="*60)

def load_cost_tracker():
    """Carrega o monitor de custos."""
    try:
        from api_cost_tracker import cost_tracker
        return cost_tracker, None
    except ImportError as e:
        return None, f"❌ Erro ao importar: {e}"
    except Exception as e:
        return None, f"❌ Erro: {e}"

def show_main_menu():
    """Mostra menu principal."""
    clear_screen()
    print_header("RELATÓRIO DE CUSTOS - CHATBOT ANALÍTICO")
    
    # Tenta carregar o tracker
    tracker, error = load_cost_tracker()
    
    if error:
        print(f"\n{error}")
        print("\n💡 Verifique se:")
        print("   1. O arquivo api_cost_tracker.py está na mesma pasta")
        print("   2. As dependências estão instaladas")
        print("   3. Você está no diretório correto")
        input("\nPressione Enter para sair...")
        return
    
    # Mostra resumo rápido
    print(f"\n📈 RESUMO RÁPIDO:")
    today = date.today().isoformat()
    daily = tracker.get_daily_summary(today)
    
    print(f"   • Data: {today}")
    print(f"   • Custo hoje: ${daily['cost']:.6f} USD")
    print(f"   • Tokens: {daily['tokens']['input']:,}+{daily['tokens']['output']:,}")
    print(f"   • Consultas hoje: {daily.get('calls_today', 0)}")
    
    total_cost = tracker.daily_stats.get("total_cost", 0.0)
    total_calls = sum(stats["calls"] for stats in tracker.daily_stats.get("model_usage", {}).values())
    print(f"   • Custo total: ${total_cost:.6f} USD")
    print(f"   • Total consultas: {total_calls}")
    
    print("\n🔧 MENU PRINCIPAL:")
    print("1. 📋 Ver relatório detalhado")
    print("2. 📁 Exportar para CSV/Excel")
    print("3. 📅 Análise por período")
    print("4. ⚠️  Configurar alertas")
    print("5. 🔍 Ver consultas recentes")
    print("6. 🗑️  Limpar dados antigos")
    print("7. 🆘 Ajuda e informações")
    print("8. 🚪 Sair")
    
    return tracker

def option_detailed_report(tracker):
    """Opção 1: Relatório detalhado."""
    clear_screen()
    print_header("RELATÓRIO DETALHADO")
    
    tracker.print_summary()
    
    input("\n📝 Pressione Enter para continuar...")

def option_export_csv(tracker):
    """Opção 2: Exportar para CSV."""
    clear_screen()
    print_header("EXPORTAR PARA CSV")
    
    print("📁 Formatos disponíveis:")
    print("1. CSV simples (api_costs_report.csv)")
    print("2. CSV com conversão para BRL")
    print("3. Excel (.xlsx)")
    print("4. JSON completo")
    
    choice = input("\nEscolha o formato (1-4): ").strip()
    
    try:
        if choice == "1":
            filename = "data/api_costs_report.csv"
            df = tracker.export_to_csv(filename)
            if df is not None:
                print(f"\n✅ Exportado: {filename}")
                print(f"   • Registros: {len(df)}")
                print(f"   • Período: {df['date'].min()} a {df['date'].max()}")
        
        elif choice == "2":
            filename = "data/api_costs_report_brl.csv"
            # Exporta com conversão para Real
            try:
                data = []
                for day, cost in tracker.daily_stats["daily_costs"].items():
                    tokens = tracker.daily_stats["daily_tokens"].get(day, {"input": 0, "output": 0})
                    data.append({
                        "data": day,
                        "custo_usd": cost,
                        "custo_brl": cost * 5.0,  # Taxa de conversão
                        "tokens_entrada": tokens["input"],
                        "tokens_saida": tokens["output"],
                        "total_tokens": tokens["input"] + tokens["output"],
                        "consultas_dia": len([c for c in tracker.daily_stats.get("detailed_calls", []) 
                                            if c["timestamp"].startswith(day)])
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df = df.sort_values("data", ascending=False)
                    df.to_csv(filename, index=False, encoding='utf-8')
                    print(f"\n✅ Exportado: {filename}")
                    print(f"   • Taxa de conversão: USD 1.00 = BRL 5.00")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        elif choice == "3":
            filename = "data/api_costs_report.xlsx"
            try:
                df = tracker.export_to_csv("data/api_costs_temp.csv")
                if df is not None:
                    # Adiciona formatação para Excel
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Custos', index=False)
                        
                        # Adiciona resumo em outra aba
                        summary_data = []
                        for model, stats in tracker.daily_stats.get("model_usage", {}).items():
                            summary_data.append({
                                "Modelo": model,
                                "Chamadas": stats["calls"],
                                "Custo_USD": stats["total_cost"],
                                "Custo_BRL": stats["total_cost"] * 5.0,
                                "Tokens_Entrada": stats["total_tokens"]["input"],
                                "Tokens_Saida": stats["total_tokens"]["output"]
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Resumo', index=False)
                    
                    print(f"\n✅ Exportado: {filename}")
                    print("   • Abas: 'Custos' e 'Resumo'")
            
            except ImportError:
                print("❌ Para exportar Excel, instale: pip install openpyxl")
        
        elif choice == "4":
            filename = "data/api_costs_full.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(tracker.daily_stats, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Exportado: {filename}")
                print(f"   • Tamanho: {os.path.getsize(filename) / 1024:.1f} KB")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        else:
            print("❌ Opção inválida")
    
    except Exception as e:
        print(f"❌ Erro ao exportar: {e}")
    
    input("\n📝 Pressione Enter para continuar...")

def option_period_analysis(tracker):
    """Opção 3: Análise por período."""
    clear_screen()
    print_header("ANÁLISE POR PERÍODO")
    
    print("📅 Escolha o período:")
    print("1. Hoje")
    print("2. Últimos 7 dias")
    print("3. Este mês")
    print("4. Mês específico")
    print("5. Período personalizado")
    
    choice = input("\nEscolha (1-5): ").strip()
    
    try:
        if choice == "1":
            daily = tracker.get_daily_summary()
            print(f"\n📊 HOJE ({daily['date']}):")
            print(f"   • Custo: ${daily['cost']:.6f}")
            print(f"   • Tokens: {daily['tokens']['input']:,}+{daily['tokens']['output']:,}")
            print(f"   • Consultas: {daily.get('calls_today', 0)}")
        
        elif choice == "2":
            print("\n📊 ÚLTIMOS 7 DIAS:")
            today = date.today()
            total_cost = 0
            total_tokens = {"input": 0, "output": 0}
            total_calls = 0
            
            for i in range(7):
                day = today - timedelta(days=i)
                day_str = day.isoformat()
                daily = tracker.get_daily_summary(day_str)
                
                if daily['cost'] > 0:
                    total_cost += daily['cost']
                    total_tokens["input"] += daily['tokens']['input']
                    total_tokens["output"] += daily['tokens']['output']
                    total_calls += daily.get('calls_today', 0)
                    
                    print(f"   • {day_str}: ${daily['cost']:.4f} | "
                          f"Tokens: {daily['tokens']['input']:,}+{daily['tokens']['output']:,}")
            
            print(f"\n📈 TOTAL 7 DIAS:")
            print(f"   • Custo: ${total_cost:.6f}")
            print(f"   • Tokens: {total_tokens['input']:,}+{total_tokens['output']:,}")
            print(f"   • Consultas: {total_calls}")
        
        elif choice == "3":
            today = datetime.now()
            monthly = tracker.get_monthly_summary(today.year, today.month)
            
            print(f"\n📊 ESTE MÊS ({today.year}/{today.month:02d}):")
            print(f"   • Custo total: ${monthly['total_cost']:.6f}")
            print(f"   • Tokens: {monthly['total_tokens']['input']:,}+{monthly['total_tokens']['output']:,}")
            print(f"   • Dias com uso: {monthly['days']}")
            
            if monthly['daily_breakdown']:
                print(f"\n📅 DETALHAMENTO DIÁRIO:")
                for day in monthly['daily_breakdown'][:10]:  # Mostra até 10 dias
                    print(f"   • {day['date']}: ${day['cost']:.4f}")
        
        elif choice == "4":
            year = input("Ano (YYYY): ").strip()
            month = input("Mês (MM): ").strip()
            
            try:
                year = int(year)
                month = int(month)
                monthly = tracker.get_monthly_summary(year, month)
                
                print(f"\n📊 MÊS {month:02d}/{year}:")
                print(f"   • Custo total: ${monthly['total_cost']:.6f}")
                print(f"   • Tokens: {monthly['total_tokens']['input']:,}+{monthly['total_tokens']['output']:,}")
                print(f"   • Dias com uso: {monthly['days']}")
            
            except ValueError:
                print("❌ Data inválida")
        
        elif choice == "5":
            start = input("Data inicial (YYYY-MM-DD): ").strip()
            end = input("Data final (YYYY-MM-DD): ").strip()
            
            try:
                start_date = datetime.strptime(start, "%Y-%m-%d")
                end_date = datetime.strptime(end, "%Y-%m-%d")
                
                total_cost = 0
                total_tokens = {"input": 0, "output": 0}
                days_count = 0
                
                current_date = start_date
                while current_date <= end_date:
                    day_str = current_date.strftime("%Y-%m-%d")
                    daily = tracker.get_daily_summary(day_str)
                    
                    if daily['cost'] > 0:
                        total_cost += daily['cost']
                        total_tokens["input"] += daily['tokens']['input']
                        total_tokens["output"] += daily['tokens']['output']
                        days_count += 1
                    
                    current_date += timedelta(days=1)
                
                print(f"\n📊 PERÍODO: {start} a {end}")
                print(f"   • Custo total: ${total_cost:.6f}")
                print(f"   • Tokens: {total_tokens['input']:,}+{total_tokens['output']:,}")
                print(f"   • Dias com uso: {days_count}")
                print(f"   • Dias totais: {(end_date - start_date).days + 1}")
            
            except ValueError:
                print("❌ Data inválida. Use formato YYYY-MM-DD")
        
        else:
            print("❌ Opção inválida")
    
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\n📝 Pressione Enter para continuar...")

def option_set_alerts(tracker):
    """Opção 4: Configurar alertas."""
    clear_screen()
    print_header("CONFIGURAR ALERTAS DE CUSTO")
    
    print("⚠️  Configure limites de custo para receber alertas:")
    print("1. Limite diário")
    print("2. Limite mensal")
    print("3. Limite por consulta")
    print("4. Ver configuração atual")
    
    choice = input("\nEscolha (1-4): ").strip()
    
    # Arquivo de configuração
    config_file = "data/cost_alerts.json"
    
    try:
        # Carrega configuração existente
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "daily_limit": 1.0,  # USD
                "monthly_limit": 20.0,  # USD
                "per_query_limit": 0.1,  # USD
                "email_alerts": False,
                "console_alerts": True,
                "last_checked": None
            }
        
        if choice == "1":
            limit = input("Limite diário (USD): ").strip()
            try:
                config["daily_limit"] = float(limit)
                print(f"✅ Limite diário definido: ${limit}")
            except ValueError:
                print("❌ Valor inválido")
        
        elif choice == "2":
            limit = input("Limite mensal (USD): ").strip()
            try:
                config["monthly_limit"] = float(limit)
                print(f"✅ Limite mensal definido: ${limit}")
            except ValueError:
                print("❌ Valor inválido")
        
        elif choice == "3":
            limit = input("Limite por consulta (USD): ").strip()
            try:
                config["per_query_limit"] = float(limit)
                print(f"✅ Limite por consulta definido: ${limit}")
            except ValueError:
                print("❌ Valor inválido")
        
        elif choice == "4":
            print("\n⚙️  CONFIGURAÇÃO ATUAL:")
            print(f"   • Limite diário: ${config['daily_limit']:.2f}")
            print(f"   • Limite mensal: ${config['monthly_limit']:.2f}")
            print(f"   • Limite por consulta: ${config['per_query_limit']:.2f}")
            print(f"   • Alertas no console: {'✅' if config['console_alerts'] else '❌'}")
            print(f"   • Alertas por email: {'✅' if config['email_alerts'] else '❌'}")
        
        else:
            print("❌ Opção inválida")
            return
        
        # Salva configuração
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📁 Configuração salva em: {config_file}")
        
        # Verifica se algum limite está próximo de ser atingido
        check_limits(tracker, config)
    
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\n📝 Pressione Enter para continuar...")

def check_limits(tracker, config):
    """Verifica se algum limite está próximo."""
    daily = tracker.get_daily_summary()
    today_cost = daily['cost']
    
    if today_cost > config['daily_limit'] * 0.8:  # 80% do limite
        print(f"\n⚠️  ALERTA: Custo diário ({today_cost:.4f}) está próximo do limite (${config['daily_limit']:.2f})")

def option_recent_calls(tracker):
    """Opção 5: Ver consultas recentes."""
    clear_screen()
    print_header("CONSULTAS RECENTES")
    
    try:
        detailed_calls = tracker.daily_stats.get("detailed_calls", [])
        
        if not detailed_calls:
            print("📭 Nenhuma consulta registrada ainda.")
        else:
            # Mostra as últimas 10 consultas
            recent_calls = detailed_calls[-10:][::-1]  # Mais recentes primeiro
            
            print(f"📋 ÚLTIMAS {len(recent_calls)} CONSULTAS:\n")
            
            for i, call in enumerate(recent_calls, 1):
                timestamp = call['timestamp']
                time_str = timestamp[11:19] if len(timestamp) > 10 else timestamp
                date_str = timestamp[:10] if len(timestamp) > 10 else timestamp
                
                # Extrai pergunta dos metadados
                question = call['metadata'].get('question', 'N/A') if call.get('metadata') else 'N/A'
                
                print(f"{i:2d}. ⏰ {date_str} {time_str}")
                print(f"    🤖 Modelo: {call['model']}")
                print(f"    💰 Custo: ${call['cost']:.6f}")
                print(f"    🔢 Tokens: {call['prompt_tokens']}+{call['completion_tokens']}")
                print(f"    💬 Pergunta: {question[:80]}...")
                print()
        
        # Estatísticas das consultas
        if detailed_calls:
            total_cost = sum(c['cost'] for c in detailed_calls)
            avg_cost = total_cost / len(detailed_calls)
            
            print(f"📊 ESTATÍSTICAS:")
            print(f"   • Total consultas: {len(detailed_calls)}")
            print(f"   • Custo total: ${total_cost:.6f}")
            print(f"   • Custo médio: ${avg_cost:.6f}")
            print(f"   • Primeira: {detailed_calls[0]['timestamp'][:10]}")
            print(f"   • Última: {detailed_calls[-1]['timestamp'][:10]}")
    
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\n📝 Pressione Enter para continuar...")

def option_cleanup_old_data(tracker):
    """Opção 6: Limpar dados antigos."""
    clear_screen()
    print_header("LIMPAR DADOS ANTIGOS")
    
    print("⚠️  ATENÇÃO: Esta ação não pode ser desfeita!")
    print("\nEscolha o que limpar:")
    print("1. Dados com mais de 30 dias")
    print("2. Dados com mais de 90 dias")
    print("3. Todos os dados (reset completo)")
    print("4. Manter apenas último mês")
    
    choice = input("\nEscolha (1-4): ").strip()
    
    confirm = input("\n❌ CONFIRME digitando 'SIM': ").strip()
    
    if confirm.upper() != 'SIM':
        print("Operação cancelada.")
        input("\n📝 Pressione Enter para continuar...")
        return
    
    try:
        if choice == "1":
            # Mantém dados dos últimos 30 dias
            cutoff_date = (date.today() - timedelta(days=30)).isoformat()
            clean_old_data(tracker, cutoff_date)
        
        elif choice == "2":
            # Mantém dados dos últimos 90 dias
            cutoff_date = (date.today() - timedelta(days=90)).isoformat()
            clean_old_data(tracker, cutoff_date)
        
        elif choice == "3":
            # Reset completo
            tracker.daily_stats = tracker._create_empty_stats()
            tracker._save_stats()
            print("✅ Todos os dados foram removidos.")
        
        elif choice == "4":
            # Mantém apenas mês atual
            today = date.today()
            cutoff_date = date(today.year, today.month, 1).isoformat()
            clean_old_data(tracker, cutoff_date)
            print(f"✅ Mantidos apenas dados a partir de {cutoff_date}")
        
        else:
            print("❌ Opção inválida")
    
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    input("\n📝 Pressione Enter para continuar...")

def clean_old_data(tracker, cutoff_date):
    """Remove dados anteriores à data especificada."""
    try:
        # Remove custos diários antigos
        old_days = [day for day in tracker.daily_stats["daily_costs"] if day < cutoff_date]
        for day in old_days:
            tracker.daily_stats["daily_costs"].pop(day, None)
            tracker.daily_stats["daily_tokens"].pop(day, None)
        
        # Remove chamadas detalhadas antigas
        if "detailed_calls" in tracker.daily_stats:
            tracker.daily_stats["detailed_calls"] = [
                call for call in tracker.daily_stats["detailed_calls"]
                if call["timestamp"][:10] >= cutoff_date
            ]
        
        # Recalcula totais
        total_cost = sum(tracker.daily_stats["daily_costs"].values())
        tracker.daily_stats["total_cost"] = total_cost
        
        total_input = sum(t.get("input", 0) for t in tracker.daily_stats["daily_tokens"].values())
        total_output = sum(t.get("output", 0) for t in tracker.daily_stats["daily_tokens"].values())
        tracker.daily_stats["total_tokens"] = {"input": total_input, "output": total_output}
        
        # Salva
        tracker._save_stats()
        
        print(f"✅ Dados anteriores a {cutoff_date} removidos.")
        print(f"   • Dias removidos: {len(old_days)}")
    
    except Exception as e:
        raise e

def option_help():
    """Opção 7: Ajuda."""
    clear_screen()
    print_header("AJUDA E INFORMAÇÕES")
    
    print("📚 SOBRE ESTE SISTEMA:")
    print("   • Monitora custos da API OpenAI em tempo real")
    print("   • Salva dados automaticamente em data/api_costs.json")
    print("   • Calcula custos baseado nos preços oficiais")
    
    print("\n💡 PREÇOS UTILIZADOS (USD por 1M tokens):")
    print("   • GPT-4o: Entrada $2.50 | Saída $10.00")
    print("   • GPT-4o-mini: Entrada $0.15 | Saída $0.60")
    print("   • GPT-4-turbo: Entrada $10.00 | Saída $30.00")
    print("   • GPT-3.5-turbo: Entrada $0.50 | Saída $1.50")
    
    print("\n🚀 COMO USAR:")
    print("   1. Execute: python check_costs.py")
    print("   2. Escolha uma opção do menu")
    print("   3. Use durante o desenvolvimento para monitorar custos")
    print("   4. Configure alertas para limites de gasto")
    
    print("\n📁 ARQUIVOS GERADOS:")
    print("   • data/api_costs.json - Dados brutos (automático)")
    print("   • data/api_costs_report.csv - Relatório CSV (manual)")
    print("   • data/cost_alerts.json - Configuração de alertas")
    
    print("\n🔧 INTEGRAÇÃO COM O CHATBOT:")
    print("   • Custo é registrado automaticamente a cada consulta")
    print("   • Use o botão 'Ver Custos' no frontend para dashboard")
    print("   • Endpoint API: GET /api/costs")
    
    input("\n📝 Pressione Enter para continuar...")

def main():
    """Função principal."""
    while True:
        tracker = show_main_menu()
        
        if tracker is None:
            break  # Sai se houve erro ao carregar
        
        choice = input("\nEscolha uma opção (1-8): ").strip()
        
        if choice == "1":
            option_detailed_report(tracker)
        elif choice == "2":
            option_export_csv(tracker)
        elif choice == "3":
            option_period_analysis(tracker)
        elif choice == "4":
            option_set_alerts(tracker)
        elif choice == "5":
            option_recent_calls(tracker)
        elif choice == "6":
            option_cleanup_old_data(tracker)
        elif choice == "7":
            option_help()
        elif choice == "8":
            print("\n👋 Saindo... Até logo!")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")
            input("\n📝 Pressione Enter para continuar...")

if __name__ == "__main__":
    main()