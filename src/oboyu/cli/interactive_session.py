"""Interactive query session for continuous searches."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text

from oboyu.common.types import SearchResult
from oboyu.retriever.retriever import Retriever


class InteractiveQuerySession:
    """Interactive query session for continuous searches."""

    def __init__(
        self,
        retriever: Retriever,
        initial_config: Dict[str, Any],
        console: Console,
    ) -> None:
        """Initialize the interactive session.

        Args:
            retriever: The initialized retriever instance
            initial_config: Initial configuration settings
            console: Rich console for output

        """
        self.retriever = retriever
        self.config = initial_config.copy()
        self.console = console

        # Create history directory if needed
        history_dir = Path.home() / ".oboyu"
        history_dir.mkdir(exist_ok=True)

        # Initialize prompt session with auto-suggestions and history
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_dir / "query_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                [
                    "/help",
                    "/exit",
                    "/quit",
                    "/q",
                    "/mode",
                    "/topk",
                    "/top-k",
                    "/rrf",
                    "/rerank",
                    "/settings",
                    "/clear",
                    "/stats",
                    "vector",
                    "bm25",
                    "hybrid",
                    "on",
                    "off",
                ]
            ),
        )

    def run(self) -> None:
        """Run the interactive session."""
        self.console.print("\n[bold blue]Oboyu Interactive Query Session[/bold blue]")
        self.console.print("Type [bold green]/help[/bold green] for commands or [bold red]/exit[/bold red] to quit.\n")

        while True:
            try:
                # Get user input
                query = self.session.prompt("🔍 Query: ")

                # Handle empty queries
                if not query.strip():
                    continue

                # Handle commands
                if query.startswith("/"):
                    if self._handle_command(query):
                        break  # Exit if command returned True
                    continue

                # Execute search
                self._execute_search(query)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logging.error(f"Interactive session error: {e}")

    def _handle_command(self, command: str) -> bool:
        """Handle interactive commands.

        Args:
            command: The command string

        Returns:
            True if session should exit, False otherwise

        """
        cmd_parts = command.strip().split()
        cmd = cmd_parts[0].lower()

        if cmd in ["/help", "/h"]:
            self._show_help()
        elif cmd in ["/exit", "/quit", "/q"]:
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            return True
        elif cmd in ["/mode", "/m"]:
            self._handle_mode_command(cmd_parts)
        elif cmd in ["/topk", "/top-k", "/k"]:
            self._handle_topk_command(cmd_parts)
        elif cmd in ["/rrf", "/r"]:
            self._handle_rrf_command(cmd_parts)
        elif cmd in ["/rerank", "/rerank"]:
            self._handle_rerank_command(cmd_parts)
        elif cmd in ["/settings", "/config", "/s"]:
            self._show_settings()
        elif cmd in ["/clear", "/cls"]:
            self._clear_screen()
        elif cmd in ["/stats"]:
            self._show_stats()
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type [bold green]/help[/bold green] for available commands.")

        return False

    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
[bold blue]Available Commands:[/bold blue]

[bold green]Search Commands:[/bold green]
  [cyan]/mode <vector|bm25|hybrid>[/cyan]     Set search mode
  [cyan]/topk <number>[/cyan]                 Set number of results (1-100)
  [cyan]/rrf <k>[/cyan]                     Set RRF parameter for hybrid search (10-200)
  [cyan]/rerank <on|off>[/cyan]              Enable/disable reranking

[bold green]Utility Commands:[/bold green]
  [cyan]/settings[/cyan]                      Show current settings
  [cyan]/stats[/cyan]                         Show database statistics
  [cyan]/clear[/cyan]                         Clear screen
  [cyan]/help[/cyan]                          Show this help
  [cyan]/exit[/cyan]                          Exit session

[bold green]Examples:[/bold green]
  [dim]Query example[/dim]
  [dim]/mode hybrid[/dim]
  [dim]/topk 10[/dim]
  [dim]/weights 0.7 0.3[/dim]
        """
        self.console.print(help_text)

    def _handle_mode_command(self, cmd_parts: list[str]) -> None:
        """Handle mode command."""
        if len(cmd_parts) < 2:
            self.console.print("[yellow]Usage: /mode <vector|bm25|hybrid>[/yellow]")
            return

        mode = cmd_parts[1].lower()
        if mode in ["vector", "bm25", "hybrid"]:
            self.config["mode"] = mode
            self.console.print(f"[green]Search mode set to: {mode}[/green]")
        else:
            self.console.print("[red]Invalid mode. Use: vector, bm25, or hybrid[/red]")

    def _handle_topk_command(self, cmd_parts: list[str]) -> None:
        """Handle topk command."""
        if len(cmd_parts) < 2:
            self.console.print("[yellow]Usage: /topk <number>[/yellow]")
            return

        try:
            topk = int(cmd_parts[1])
            if 1 <= topk <= 100:
                self.config["top_k"] = topk
                self.console.print(f"[green]Top-k set to: {topk}[/green]")
            else:
                self.console.print("[red]Top-k must be between 1 and 100[/red]")
        except ValueError:
            self.console.print("[red]Invalid number format[/red]")

    def _handle_rrf_command(self, cmd_parts: list[str]) -> None:
        """Handle RRF parameter command."""
        if len(cmd_parts) < 2:
            self.console.print("[yellow]Usage: /rrf <k_value>[/yellow]")
            self.console.print("[dim]Example: /rrf 60 (default), /rrf 30 (more aggressive), /rrf 100 (more conservative)[/dim]")
            return

        try:
            rrf_k = int(cmd_parts[1])
            
            if 1 <= rrf_k <= 1000:  # Reasonable range for RRF k parameter
                self.config["rrf_k"] = rrf_k
                self.console.print(f"[green]RRF parameter set to: k={rrf_k}[/green]")
                if rrf_k < 30:
                    self.console.print("[yellow]Low k value - more aggressive fusion of top results[/yellow]")
                elif rrf_k > 100:
                    self.console.print("[yellow]High k value - more conservative, balanced fusion[/yellow]")
            else:
                self.console.print("[red]RRF k parameter must be between 1 and 1000 (typical range: 10-200)[/red]")
        except ValueError:
            self.console.print("[red]Invalid RRF parameter format - must be an integer[/red]")

    def _handle_rerank_command(self, cmd_parts: list[str]) -> None:
        """Handle rerank command."""
        if len(cmd_parts) < 2:
            self.console.print("[yellow]Usage: /rerank <on|off>[/yellow]")
            return

        setting = cmd_parts[1].lower()
        if setting in ["on", "true", "1"]:
            # Check if reranker is available
            if not self.retriever.reranker_service or not self.retriever.reranker_service.is_available():
                self.console.print("[yellow]Warning: Reranker service is not available. Enable reranker in config first.[/yellow]")
            self.config["rerank"] = True
            self.console.print("[green]Reranking enabled[/green]")
        elif setting in ["off", "false", "0"]:
            self.config["rerank"] = False
            self.console.print("[green]Reranking disabled[/green]")
        else:
            self.console.print("[red]Use 'on' or 'off'[/red]")

    def _show_settings(self) -> None:
        """Show current settings."""
        self.console.print("\n[bold blue]Current Settings:[/bold blue]")
        for key, value in self.config.items():
            self.console.print(f"  [cyan]{key}[/cyan]: {value}")
        
        # Show reranker availability
        reranker_available = (
            self.retriever.reranker_service and
            self.retriever.reranker_service.is_available()
        )
        self.console.print(f"  [cyan]reranker_available[/cyan]: {reranker_available}")
        self.console.print()

    def _clear_screen(self) -> None:
        """Clear the screen."""
        # Use ANSI escape codes for cross-platform screen clearing
        import os
        
        if os.name == 'nt':
            # Windows - use ANSI codes if supported, otherwise print newlines
            print('\033[2J\033[H', end='')
        else:
            # Unix-like systems - use ANSI escape codes
            print('\033[2J\033[H', end='')

    def _show_stats(self) -> None:
        """Show database statistics."""
        try:
            stats = self.retriever.get_database_stats()
            self.console.print("\n[bold blue]Database Statistics:[/bold blue]")
            
            # Format basic stats
            chunk_count = stats.get('chunk_count', 0)
            embedding_count = stats.get('embedding_count', 0)
            vocabulary_size = stats.get('vocabulary_size', 0)
            
            self.console.print(f"  [cyan]Chunks[/cyan]: {chunk_count:,}")
            self.console.print(f"  [cyan]Embeddings[/cyan]: {embedding_count:,}")
            self.console.print(f"  [cyan]Vocabulary[/cyan]: {vocabulary_size:,}")
            
            # Show languages if available
            if 'languages' in stats and stats['languages']:
                languages = stats['languages'].split(',') if isinstance(stats['languages'], str) else stats['languages']
                self.console.print(f"  [cyan]Languages[/cyan]: {', '.join(languages)}")
            
            # Show model info
            if 'embedding_model' in stats and stats['embedding_model']:
                self.console.print(f"  [cyan]Model[/cyan]: {stats['embedding_model']}")
            
            self.console.print()
            
        except Exception as e:
            self.console.print(f"[red]Error getting stats: {e}[/red]")

    def _execute_search(self, query: str) -> None:
        """Execute a search query."""
        try:
            start_time = time.time()
            
            # Create search parameters
            search_params = {
                "top_k": self.config.get("top_k", 10),
            }
            
            # Add mode-specific parameters
            mode = self.config.get("mode", "hybrid")
            # Note: RRF parameter is configured at the indexer level, not passed as search parameter
            
            # Execute search based on mode
            if mode == "vector":
                results = self.retriever.vector_search(query, **search_params)
            elif mode == "bm25":
                results = self.retriever.bm25_search(query, **search_params)
            else:  # hybrid
                results = self.retriever.hybrid_search(query, **search_params)
            
            # Apply reranking if enabled and available
            reranker_used = False
            if (self.config.get("rerank", False) and results and
                self.retriever.reranker_service and
                self.retriever.reranker_service.is_available()):
                results = self.retriever.rerank_results(query, results)
                reranker_used = True
            
            elapsed_time = time.time() - start_time
            
            # Display results
            self._display_results(results, elapsed_time, mode, reranker_used)
            
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
            logging.error(f"Search error: {e}")

    def _display_results(
        self,
        results: list[SearchResult],
        elapsed_time: float,
        mode: str,
        reranker_used: bool = False
    ) -> None:
        """Display search results."""
        if not results:
            self.console.print("[yellow]No results found.[/yellow]")
            return
        
        # Header with reranker indication
        reranker_suffix = " with reranker" if reranker_used else ""
        self.console.print(f"\n[bold green]Found {len(results)} results[/bold green] ([dim]{mode} search{reranker_suffix}, {elapsed_time:.3f}s[/dim])\n")
        
        # Results
        for i, result in enumerate(results, 1):
            # Score with color coding
            score = result.score
            if score >= 0.8:
                score_color = "bright_green"
            elif score >= 0.6:
                score_color = "green"
            elif score >= 0.4:
                score_color = "yellow"
            else:
                score_color = "red"
            
            # File path (shortened)
            file_path = str(result.path)
            if len(file_path) > 60:
                file_path = "..." + file_path[-57:]
            
            # Header line
            self.console.print(f"[bold blue]{i:2d}.[/bold blue] [{score_color}]{score:.3f}[/{score_color}] [dim]{file_path}[/dim]")
            
            # Title if available
            if result.title and result.title.strip():
                title_text = Text(result.title.strip())
                title_text.stylize("bold")
                self.console.print(f"    {title_text}")
            
            # Content preview
            content_preview = result.content[:200].replace('\n', ' ').strip()
            if len(result.content) > 200:
                content_preview += "..."
            
            self.console.print(f"    {content_preview}")
            self.console.print()  # Empty line between results
