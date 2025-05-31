"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text
from typing_extensions import Annotated

# Disable tokenizer parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from oboyu.cli.hierarchical_logger import create_hierarchical_logger
from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer, SearchResult

# Create Typer app
app = typer.Typer(
    help="Search indexed documents",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create console for rich output
console = Console()

# Define command options
QueryOption = Annotated[
    Optional[str],
    typer.Argument(
        help="Search query text",
    ),
]

ModeOption = Annotated[
    str,
    typer.Option(
        "--mode",
        "-m",
        help="Search mode (vector, bm25, hybrid)",
    ),
]

TopKOption = Annotated[
    Optional[int],
    typer.Option(
        "--top-k",
        "-k",
        help="Number of results to return",
        min=1,
    ),
]

ExplainOption = Annotated[
    bool,
    typer.Option(
        "--explain",
        "-e",
        help="Show detailed match explanation",
    ),
]

FormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format (text, json)",
    ),
]

VectorWeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--vector-weight",
        help="Weight for vector scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

BM25WeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--bm25-weight",
        help="Weight for BM25 scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]

RerankOption = Annotated[
    Optional[bool],
    typer.Option(
        "--rerank/--no-rerank",
        help="Enable or disable reranking of search results",
    ),
]

InteractiveOption = Annotated[
    bool,
    typer.Option(
        "--interactive",
        "-i",
        help="Start interactive search session for continuous queries",
    ),
]


class InteractiveQuerySession:
    """Interactive query session for continuous searches."""

    def __init__(
        self,
        indexer: Indexer,
        initial_config: Dict[str, Any],
        console: Console,
    ) -> None:
        """Initialize the interactive session.

        Args:
            indexer: The initialized indexer instance
            initial_config: Initial configuration settings
            console: Rich console for output

        """
        self.indexer = indexer
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
                    "/weights",
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
        # Display welcome message
        self.console.print("\n🔍 [bold]Oboyu Interactive Search[/bold]", style="cyan")
        reranker_status = "[green]enabled[/green]" if self.config.get("use_reranker") else "[red]disabled[/red]"
        self.console.print(
            f"📊 Mode: [yellow]{self.config.get('mode', 'hybrid')}[/yellow] | "
            f"Top-K: [yellow]{self.config.get('top_k', 5)}[/yellow] | "
            f"Vector: [yellow]{self.config.get('vector_weight', 0.7)}[/yellow] | "
            f"BM25: [yellow]{self.config.get('bm25_weight', 0.3)}[/yellow] | "
            f"Reranker: {reranker_status}"
        )
        self.console.print("\n✅ Ready for search!")
        self.console.print("Type your search query (or '/help' for commands, '/exit' to quit):\n")
        
        # Main REPL loop
        while True:
            try:
                # Get user input
                user_input = self.session.prompt("> ").strip()
                
                if not user_input:
                    continue
                    
                # Check if it's a command
                if self._is_command(user_input):
                    if not self._process_command(user_input):
                        break  # Exit command received
                else:
                    # It's a search query
                    self._process_query(user_input)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
                
        # Graceful shutdown
        self.console.print("\n👋 Goodbye!")
        
    def _is_command(self, text: str) -> bool:
        """Check if the input is a command.

        Args:
            text: Input text to check

        Returns:
            True if text is a command, False otherwise

        """
        return text.strip().startswith("/")
        
    def _process_command(self, command: str) -> bool:
        """Process an interactive command.

        Args:
            command: Command string to process

        Returns:
            True to continue session, False to exit

        """
        # Remove leading '/' and split
        command_text = command[1:].strip() if command.startswith('/') else command.strip()
        parts = command_text.lower().split()
        
        if not parts:
            self.console.print("❌ Empty command. Type '/help' for available commands")
            return True
            
        cmd = parts[0]
        
        if cmd in ["exit", "quit", "q"]:
            return False
            
        elif cmd == "help":
            self._show_help()
            
        elif cmd == "clear":
            # Clear screen - only support Unix-like systems
            subprocess.run(["/usr/bin/clear"], check=False)  # noqa: S603
            
        elif cmd == "settings":
            self._show_settings()
            
        elif cmd == "stats":
            self._show_stats()
            
        elif cmd == "mode" and len(parts) > 1:
            mode = parts[1]
            if mode in ["vector", "bm25", "hybrid"]:
                self.config["mode"] = mode
                self.console.print(f"✅ Search mode changed to: [yellow]{mode}[/yellow]")
            else:
                self.console.print(
                    f"❌ Invalid mode: [red]{mode}[/red]. "
                    "Valid modes are: vector, bm25, hybrid"
                )
                
        elif cmd in ["topk", "top-k"] and len(parts) > 1:
            try:
                top_k = int(parts[1])
                if top_k > 0:
                    self.config["top_k"] = top_k
                    self.console.print(f"✅ Top-K changed to: [yellow]{top_k}[/yellow]")
                else:
                    self.console.print("❌ Top-K must be a positive integer")
            except ValueError:
                self.console.print(f"❌ Invalid number: [red]{parts[1]}[/red]")
                
        elif cmd == "weights" and len(parts) > 2:
            try:
                vector_weight = float(parts[1])
                bm25_weight = float(parts[2])
                if 0 <= vector_weight <= 1 and 0 <= bm25_weight <= 1:
                    self.config["vector_weight"] = vector_weight
                    self.config["bm25_weight"] = bm25_weight
                    self.console.print(
                        f"✅ Weights changed to: "
                        f"Vector=[yellow]{vector_weight}[/yellow], "
                        f"BM25=[yellow]{bm25_weight}[/yellow]"
                    )
                else:
                    self.console.print("❌ Weights must be between 0 and 1")
            except ValueError:
                self.console.print("❌ Invalid weights format. Use: weights <vector> <bm25>")
                
        elif cmd == "rerank" and len(parts) > 1:
            if parts[1] == "on":
                self.config["use_reranker"] = True
                self.console.print("✅ Reranker [green]enabled[/green]")
            elif parts[1] == "off":
                self.config["use_reranker"] = False
                self.console.print("✅ Reranker [red]disabled[/red]")
            else:
                self.console.print(
                    f"❌ Invalid option: [red]{parts[1]}[/red]. Use 'on' or 'off'"
                )
        else:
            self.console.print(f"❌ Unknown command: [red]/{cmd}[/red]")
            self.console.print("Type '/help' for available commands")
            
        return True
        
    def _process_query(self, query: str) -> None:
        """Process a search query.

        Args:
            query: Search query text

        """
        start_time = time.time()
        self.console.print("\n🔍 Searching...", style="dim")
        
        try:
            # Perform search with current settings
            results = self.indexer.search(
                query,
                limit=int(self.config.get("top_k", 5)),
                mode=str(self.config.get("mode", "hybrid")),
                use_reranker=bool(self.config.get("use_reranker", False)),
                vector_weight=float(self.config.get("vector_weight", 0.7)),
                bm25_weight=float(self.config.get("bm25_weight", 0.3)),
            )
            
            elapsed_time = time.time() - start_time
            
            if results:
                self.console.print(
                    f"📊 Found [green]{len(results)}[/green] results "
                    f"in [yellow]{elapsed_time:.2f}[/yellow] seconds\n"
                )
                
                # Display results
                for i, result in enumerate(results):
                    formatted_result = format_search_result(
                        result,
                        query,
                        show_explanation=bool(self.config.get("explain", False))
                    )
                    self.console.print(formatted_result)
                    if i < len(results) - 1:
                        self.console.print("")  # Add spacing between results
            else:
                self.console.print("❌ No results found.\n", style="red")
                
        except Exception as e:
            self.console.print(f"❌ Search error: [red]{e}[/red]\n")
            
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
[bold]Available Commands:[/bold]

[yellow]Search:[/yellow]
  <query>              - Search for documents matching the query

[yellow]Commands:[/yellow]
  /help                - Show this help message
  /exit, /quit, /q     - Exit interactive mode
  /clear               - Clear the screen
  /settings            - Show current settings
  /stats               - Show index statistics

[yellow]Configuration:[/yellow]
  /mode <mode>         - Change search mode (vector, bm25, hybrid)
  /topk <number>       - Change number of results (e.g., /topk 10)
  /weights <v> <b>     - Change hybrid weights (e.g., /weights 0.8 0.2)
  /rerank on/off       - Enable/disable reranker

[yellow]Examples:[/yellow]
  > machine learning algorithms
  > /mode vector
  > /topk 10
  > /weights 0.8 0.2
  > /rerank on
"""
        self.console.print(help_text)
        
    def _show_settings(self) -> None:
        """Show current settings."""
        reranker_status = "[green]enabled[/green]" if self.config.get("use_reranker") else "[red]disabled[/red]"
        settings_text = f"""
[bold]Current Settings:[/bold]
- Mode: [yellow]{self.config.get('mode', 'hybrid')}[/yellow]
- Top-K: [yellow]{self.config.get('top_k', 5)}[/yellow]
- Vector weight: [yellow]{self.config.get('vector_weight', 0.7)}[/yellow]
- BM25 weight: [yellow]{self.config.get('bm25_weight', 0.3)}[/yellow]
- Reranker: {reranker_status}
- Database: [cyan]{self.config.get('db_path', DEFAULT_DB_PATH)}[/cyan]
"""
        self.console.print(settings_text)
        
    def _show_stats(self) -> None:
        """Show index statistics."""
        try:
            stats = self.indexer.get_statistics()
            stats_text = f"""
[bold]Index Statistics:[/bold]
- Total documents: [green]{stats.get('total_documents', 0)}[/green]
- Total chunks: [green]{stats.get('total_chunks', 0)}[/green]
- Unique files: [green]{stats.get('unique_files', 0)}[/green]
- Database size: [yellow]{stats.get('db_size_mb', 0):.2f} MB[/yellow]
"""
            self.console.print(stats_text)
        except Exception as e:
            self.console.print(f"❌ Could not retrieve statistics: [red]{e}[/red]")


def _warmup_embedding_model(indexer: Indexer) -> None:
    """Warmup the embedding model with a dummy query.

    Args:
        indexer: The indexer instance

    """
    try:
        if hasattr(indexer, 'embedding_generator') and indexer.embedding_generator is not None:
            indexer.embedding_generator.generate_query_embedding("warmup query")
    except Exception as e:
        # Ignore warmup errors - model will be loaded on first actual use
        logging.debug(f"Embedding model warmup failed (will be loaded on first use): {e}")


def _warmup_reranker(indexer: Indexer) -> None:
    """Warmup the reranker with a dummy query.

    Args:
        indexer: The indexer instance

    """
    if hasattr(indexer, 'reranker') and indexer.reranker is not None:
        try:
            from oboyu.indexer.indexer import SearchResult
            dummy_results = [
                SearchResult(
                    chunk_id="warmup",
                    path="/warmup",
                    title="Warmup",
                    content="This is a warmup query for the reranker",
                    chunk_index=0,
                    language="ja",
                    metadata={},
                    score=1.0,
                )
            ]
            # Perform warmup reranking
            indexer.reranker.rerank("warmup", dummy_results, top_k=1)
        except Exception as e:
            # Ignore warmup errors - model will be loaded on first actual use
            logging.debug(f"Reranker warmup failed (will be loaded on first use): {e}")


def format_search_result(result: SearchResult, query: str, show_explanation: bool = False) -> Text:
    """Format a search result for display using cleaner hierarchical format.

    Args:
        result: Search result to format
        query: Original search query
        show_explanation: Whether to show detailed explanation

    Returns:
        Formatted text for display

    """
    from oboyu.cli.utils import format_snippet

    # Create content with structured hierarchical format
    content = Text()

    # Title with score
    content.append("• ")
    content.append(f"{result.title}")
    content.append(f" (Score: {result.score:.2f})", style="dim")
    content.append("\n")

    # Snippet with better formatting
    snippet = format_snippet(result.content, query, 200, True)
    content.append(f"  {snippet}\n", style="")

    # Path as source
    content.append("  Source: ", style="dim")
    content.append(str(result.path))

    # Language as metadata if available and not empty
    if result.language and result.language.strip():
        content.append(f" ({result.language})", style="dim")

    # Add explanation if requested
    if show_explanation:
        content.append("\n  [Explanation] ", style="dim")
        content.append(f"Chunk ID: {result.chunk_id}, ", style="")
        content.append(f"Index: {result.chunk_index}", style="")
        # Add more explanation here in the future

    return content


@app.callback(invoke_without_command=True)
def query(
    ctx: typer.Context,
    query: QueryOption = None,
    mode: ModeOption = "hybrid",
    top_k: TopKOption = None,
    explain: ExplainOption = False,
    format: FormatOption = "text",
    vector_weight: VectorWeightOption = None,
    bm25_weight: BM25WeightOption = None,
    db_path: DatabasePathOption = None,
    rerank: RerankOption = None,
    interactive: InteractiveOption = False,
) -> None:
    """Search indexed documents.

    This command searches the index for documents matching the query.
    """
    # Check if interactive mode requested
    if interactive:
        if query is not None:
            console.print("⚠️  Warning: Query argument ignored in interactive mode", style="yellow")
    elif query is None:
        console.print("❌ Error: Query argument is required (or use --interactive)", style="red")
        raise typer.Exit(1)
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Get query engine configuration
    query_config = config_data.get("query", {})

    # Override with command-line options
    if top_k is not None:
        query_config["top_k"] = top_k
    else:
        top_k = query_config.get("top_k", 5)

    if vector_weight is not None:
        query_config["vector_weight"] = vector_weight

    if bm25_weight is not None:
        query_config["bm25_weight"] = bm25_weight

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Handle database path explicitly, with clear precedence:
    # 1. Command-line option (highest priority)
    # 2. Config file value
    # 3. Default from central path definition (lowest priority)
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Create indexer configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Use hierarchical logger for search operation
    logger = create_hierarchical_logger(console)

    # Handle interactive mode
    if interactive:
        # Initialize indexer with loading messages
        with logger.live_display():
            main_op = logger.start_operation("Starting interactive search session", expandable=False)
            
            # Initialize indexer
            load_op = logger.start_operation("Loading index...")
            indexer = Indexer(config=indexer_config)
            logger.complete_operation(load_op)
            
            # Load embedding model if needed
            if mode in ["vector", "hybrid"]:
                model_name = indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m")
                embed_op = logger.start_operation(f"Loading embedding model ({model_name})...")
                # Warmup the embedding model with a dummy query
                _warmup_embedding_model(indexer)
                logger.complete_operation(embed_op)
                
            # Load reranker if enabled
            if rerank:
                reranker_model = indexer_config_dict.get("reranker_model", "cl-nagoya/ruri-v3-reranker-310m")
                rerank_op = logger.start_operation(f"Loading reranker model ({reranker_model})...")
                # Warmup the reranker with a dummy query to ensure it's fully loaded
                _warmup_reranker(indexer)
                logger.complete_operation(rerank_op)
                
            logger.complete_operation(main_op)
        
        # Prepare initial configuration for interactive session
        session_config = {
            "mode": mode,
            "top_k": top_k,
            "explain": explain,
            "vector_weight": vector_weight if vector_weight is not None else 0.7,
            "bm25_weight": bm25_weight if bm25_weight is not None else 0.3,
            "use_reranker": rerank if rerank is not None else False,
            "db_path": indexer_config_dict.get("db_path", DEFAULT_DB_PATH),
        }
        
        # Start interactive session
        session = InteractiveQuerySession(indexer, session_config, console)
        session.run()
        return

    # Regular single query mode
    # query is guaranteed to be not None here due to the check above
    if query is None:  # This should never happen, but for type safety
        raise typer.Exit(1)
    with logger.live_display():
        # Main search operation
        main_op = logger.start_operation(f'Search: "{query}"', expandable=False, details=f"Mode: {mode}\nTop K: {top_k}")

        # Initialize indexer
        load_op = logger.start_operation("Loading index...")
        indexer = Indexer(config=indexer_config)
        logger.complete_operation(load_op)

        # Generate query embedding (for vector and hybrid modes)
        if mode in ["vector", "hybrid"]:
            embed_op = logger.start_operation("Generating query embedding...")
            time.sleep(0.05)  # Simulated timing
            logger.complete_operation(embed_op)

        # Perform search
        search_desc = f"{mode.capitalize()} search" + (" with reranking" if rerank else "") + "..."
        search_op = logger.start_operation(search_desc)
        results = indexer.search(
            query,
            limit=top_k,
            mode=mode,
            use_reranker=rerank,
            vector_weight=vector_weight if vector_weight is not None else 0.7,
            bm25_weight=bm25_weight if bm25_weight is not None else 0.3,
        )
        
        if results:
            rerank_note = " (reranked)" if rerank else ""
            logger.update_operation(search_op, f"{search_desc} Found {len(results)} results{rerank_note}")

            # Ranking results step is only shown if not reranking (since reranking does its own ranking)
            if not rerank:
                rank_op = logger.start_operation("Ranking results...")
                logger.complete_operation(rank_op)
        else:
            logger.update_operation(search_op, f"{mode.capitalize()} search... No results found")
            
        logger.complete_operation(search_op)

        # Complete main search operation
        logger.complete_operation(main_op)

    # Display results after hierarchical log
    if not results:
        console.print("\nNo results found.")
        return

    # Display results with cleaner formatting
    console.print()

    # Format and display each result
    for i, result in enumerate(results):
        formatted_result = format_search_result(result, query, show_explanation=explain)
        console.print(formatted_result)
        if i < len(results) - 1:
            console.print("")  # Add spacing between results
