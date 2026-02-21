#!/usr/bin/env python3
"""
Gourav LLM Agent - Interactive CLI
FunctionGemma (offline) for memory tasks + Gemini (cloud) for all chat
"""

import sys
from colorama import init, Fore, Style
from agent import HybridAgent
from categories import get_all_categories

init(autoreset=True)


def print_header():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🤖  Gourav LLM Agent")
    print(f"{Fore.CYAN}    FunctionGemma (chat + tools) | HNSW RAG | Gemini (online only)")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


def print_help():
    help_text = f"""
{Fore.YELLOW}💬 Chat:{Style.RESET_ALL}
  Just type — FunctionGemma answers using RAG context + memory

{Fore.YELLOW}📝 Memory Commands:{Style.RESET_ALL}
  {Fore.GREEN}/remember <fact>{Style.RESET_ALL}     - Manually add a fact
  {Fore.GREEN}/forget <pattern>{Style.RESET_ALL}    - Forget memories
  {Fore.GREEN}/categories{Style.RESET_ALL}          - Show all categories
  {Fore.GREEN}/category <name>{Style.RESET_ALL}     - View category memories
  {Fore.GREEN}/stats{Style.RESET_ALL}               - Memory & usage statistics
  {Fore.GREEN}/decay{Style.RESET_ALL}               - Run memory decay

{Fore.YELLOW}📚 Document Commands:{Style.RESET_ALL}
  {Fore.GREEN}/ingest{Style.RESET_ALL}              - Re-scan memory/ folder and index new files
  {Fore.GREEN}/ingest <path>{Style.RESET_ALL}       - Ingest a specific file
  {Fore.GREEN}/web <question>{Style.RESET_ALL}      - Force online search via Gemini+Google

{Fore.YELLOW}💬 Session Commands:{Style.RESET_ALL}
  {Fore.GREEN}/end{Style.RESET_ALL}                 - End session & commit to memory
  {Fore.GREEN}/new{Style.RESET_ALL}                 - Start new session

{Fore.YELLOW}⚙️  System Commands:{Style.RESET_ALL}
  {Fore.GREEN}/model <name>{Style.RESET_ALL}        - Change Gemini model
  {Fore.GREEN}/help{Style.RESET_ALL}                - Show this help
  {Fore.GREEN}/quit{Style.RESET_ALL}                - Exit

{Fore.YELLOW}✨ How it works:{Style.RESET_ALL}
  ⚡ FunctionGemma  — answers questions using retrieved context (offline)
  ⚡ FunctionGemma  — fact extraction, classification, conflict detection
  📚 HNSW RAG       — any file dropped in memory/ is auto-indexed
  🌐 Gemini+Search  — live queries only (news, scores, weather)

{Fore.YELLOW}📂 Drop any file into memory/ to make it searchable:{Style.RESET_ALL}
  Supported: .pdf  .txt  .md  .docx
"""
    print(help_text)


def cleanup_and_exit(agent):
    print(f"\n{Fore.YELLOW}💾 Saving session to memory...{Style.RESET_ALL}")
    try:
        session_facts = agent.session_manager.get_session_facts()
        if session_facts:
            agent.end_session_and_commit(silent=True)
            print(f"{Fore.GREEN}✅ {len(session_facts)} facts saved{Style.RESET_ALL}")
        else:
            agent.session_manager.end_current_session()
            print(f"{Fore.CYAN}Session ended{Style.RESET_ALL}")
    except Exception:
        pass
    print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}\n")


def main():
    print_header()

    print(f"{Fore.YELLOW}Initializing...{Style.RESET_ALL}")
    try:
        agent = HybridAgent(
            gemini_model="gemini-2.5-flash",
            function_model="functiongemma"
        )

        agent.start_file_watcher()

        print(f"{Fore.GREEN}✅ Ready!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   ⚡ FunctionGemma  — retrieves context from files & memory (offline){Style.RESET_ALL}")
        print(f"{Fore.CYAN}   🟡 Gemini          — generates the final answer using retrieved context{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   📚 HNSW RAG        — ChromaDB document + chat history index{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type /help or start chatting!\n{Style.RESET_ALL}")

        stats = agent.get_memory_stats()
        if stats['total'] > 0:
            print(f"{Fore.CYAN}💾 Memory loaded: {stats['shortterm']} short-term, "
                  f"{stats['longterm']} long-term{Style.RESET_ALL}")
            print(f"{Fore.CYAN}   Categories: {', '.join(stats['by_category'].keys())}{Style.RESET_ALL}\n")

    except Exception as e:
        print(f"{Fore.RED}❌ Failed to initialize: {e}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Make sure Ollama is running: ollama serve{Style.RESET_ALL}")
        sys.exit(1)

    while True:
        try:
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()

            if not user_input:
                continue

            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["/quit", "/exit"]:
                    cleanup_and_exit(agent)
                    break

                elif command == "/help":
                    print_help()

                elif command == "/remember":
                    if args:
                        print(f"{Fore.GREEN}{agent.remember(args)}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Usage: /remember <fact>{Style.RESET_ALL}")

                elif command == "/forget":
                    if args:
                        print(f"{Fore.GREEN}{agent.forget(args)}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Usage: /forget <pattern>{Style.RESET_ALL}")

                elif command == "/categories":
                    stats = agent.get_memory_stats()
                    print(f"\n{Fore.YELLOW}📂 Memory by Category:{Style.RESET_ALL}\n")
                    for category in get_all_categories():
                        count = stats['by_category'].get(category, 0)
                        if count > 0:
                            print(f"  {Fore.CYAN}{category:15s} {count:3d} memories{Style.RESET_ALL}")
                    print(f"\n{Fore.CYAN}Personal facts : {stats['total']}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Document chunks: {stats.get('document_chunks', 0)} (RAG){Style.RESET_ALL}\n")

                elif command == "/category":
                    if args:
                        print(f"\n{agent.get_category_memories(args)}\n")
                    else:
                        print(f"{Fore.RED}Usage: /category <name>{Style.RESET_ALL}")

                elif command == "/stats":
                    stats = agent.get_memory_stats()
                    doc   = stats.get('document_chunks', 0)
                    chat  = stats.get('chat_chunks', 0)
                    print(f"\n{Fore.YELLOW}📊 Statistics:{Style.RESET_ALL}\n")
                    print(f"  Personal facts  : {stats['total']} ({stats['shortterm']} short-term, {stats['longterm']} long-term)")
                    print(f"  Document chunks : {doc - chat} (RAG — files)")
                    print(f"  Chat chunks     : {chat} (RAG — past conversations, async)")
                    print(f"  Pending writes  : {stats.get('pending_writes', 0)}")
                    print(f"  Local responses : {stats['local_responses']}")
                    print(f"  Online responses: {stats['online_responses']}")
                    print(f"\n{Fore.YELLOW}By Category:{Style.RESET_ALL}")
                    for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
                        print(f"    {cat:15s} {count:3d}")
                    print()

                elif command == "/decay":
                    print(f"\n{Fore.YELLOW}Running decay...{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}{agent.run_decay()}{Style.RESET_ALL}\n")

                elif command == "/end":
                    print(f"\n{Fore.YELLOW}Ending session...{Style.RESET_ALL}\n")
                    print(f"{Fore.GREEN}{agent.end_session_and_commit()}{Style.RESET_ALL}\n")

                elif command == "/new":
                    print(f"\n{Fore.YELLOW}Starting new session...{Style.RESET_ALL}\n")
                    print(f"{Fore.GREEN}{agent.end_session_and_commit()}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}🆕 New session started!{Style.RESET_ALL}\n")

                elif command == "/web":
                    if args:
                        print(f"\n🌐 [Gemini + Search] ", end="", flush=True)
                        resp = agent._answer_with_gemini_web(args, stream=True)
                        print()
                    else:
                        print(f"{Fore.RED}Usage: /web <question>{Style.RESET_ALL}")

                elif command == "/ingest":
                    from process_pdf import ingest, ingest_folder
                    import os
                    target = args.strip() if args else "./memory"
                    if os.path.isdir(target):
                        ingest_folder(target)
                    elif os.path.isfile(target):
                        vs = agent.memory_manager.vector_store
                        n  = ingest(target, vs=vs)
                        print(f"{Fore.GREEN}✅ {n} chunks indexed{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Not found: {target}{Style.RESET_ALL}")
                    stats = agent.get_memory_stats()
                    print(f"{Fore.CYAN}Document chunks in index: {stats.get('document_chunks', 0)}{Style.RESET_ALL}\n")

                elif command == "/model":
                    if args:
                        print(agent.change_model(args))
                    else:
                        print(f"{Fore.RED}Usage: /model <model_name>{Style.RESET_ALL}")

                else:
                    print(f"{Fore.RED}Unknown command. Type /help{Style.RESET_ALL}")

            else:
                debug_mode = user_input.startswith("DEBUG:")
                if debug_mode:
                    user_input = user_input[6:].strip()
                agent.chat(user_input, stream=True, auto_extract_facts=False, debug=debug_mode)
                print()

        except KeyboardInterrupt:
            print()
            cleanup_and_exit(agent)
            break

        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
