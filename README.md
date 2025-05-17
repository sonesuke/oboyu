# Oboyu (覚ゆ)

> A Japanese-enhanced semantic search system for your local documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## What is Oboyu?

**Oboyu** (覚ゆ - meaning "to remember" or "to memorize" in ancient Japanese) is a semantic search system that creates an intelligent index of your text-based documents. Oboyu indexes directories of text files, embeds their content using vector representations, and retrieves the most relevant documents for your queries.

With specialized support for Japanese language documents, Oboyu excels at processing content in both Japanese and English, making it an ideal solution for multilingual document collections.

The system provides both command-line interface for direct queries and an MCP server mode for network-accessible document search. While Oboyu works with any text-based documents, it is particularly optimized for Japanese language content, making it unique among semantic search tools.

![Oboyu Concept](docs/images/oboyu_concept.png)

## Key Features

- **Local Directory Processing**: Index any directory of text-based documents on your local system
- **Text Format Support**: Process plain text, markdown, code files, configuration files, and more
- **Japanese Language Excellence**: First-class support for Japanese text with built-in specialized tokenization
- **Semantic Search**: Retrieve the most relevant documents using vector embeddings
- **Document-Focused Results**: Get top matching documents with URIs, titles, and relevant snippets
- **MCP Server Mode**: Run as a server with stdio interface for integration with other tools
- **Privacy-Focused**: Your documents stay on your machine - no data sent to external services by default

## Installation

```bash
# Install from PyPI
pip install oboyu

# Or install from source
git clone https://github.com/sonesuke/oboyu.git
cd oboyu
pip install -e .
```

## Quick Start

```bash
# Index a directory
oboyu index /path/to/your/documents

# Query your documents in Japanese (returns top matching documents with snippets)
oboyu query "ドキュメント内の重要な概念は何ですか？"

# Query in English is also supported
oboyu query "What are the key concepts in the documents?"

# Start the MCP server in stdio mode
oboyu mcp
```

## Configuration

Create a configuration file at `~/.oboyu/config.yaml`:

```yaml
# Basic configuration
embedding_model: "intfloat/multilingual-e5-large"
top_k: 5  # Number of results to return
  
# Processing settings
chunk_size: 512
chunk_overlap: 50
```

See the [configuration documentation](docs/configuration.md) for more options.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Configuration Options](docs/configuration.md)
- [API Reference](docs/api.md)
- [Japanese Support Details](docs/japanese.md)
- [Architecture Overview](docs/architecture.md)
- [MCP Server Guide](docs/mcp_server.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The name "Oboyu" (覚ゆ) comes from ancient Japanese, meaning "to remember" or "to memorize"
- Thanks to the open-source Japanese NLP community for their excellent tools and resources
- Built with the inspiration of making knowledge more accessible regardless of language
