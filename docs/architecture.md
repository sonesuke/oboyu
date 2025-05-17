## Architecture

Oboyu has a carefully designed architecture with specific technology choices:

- **Vector Database**: Built-in optimized vector storage system
- **Japanese Tokenization**: Integrated specialized tokenizers for Japanese text
- **Embedding Models**: Support for multiple embedding models with multilingual capabilities
- **MCP Interface**: Standard stdio-based interface for integration with other tools

These architectural decisions are built into the system for optimal performance and simplicity, removing the need for users to configure complex underlying components.## Japanese Language Support

Oboyu provides exceptional support for Japanese language documents:

- Automatic encoding detection for Japanese text files (UTF-8, Shift-JIS, EUC-JP)
- Built-in specialized Japanese tokenization
- Support for mixed Japanese-English content
- Japanese-optimized embedding models
- Accurate retrieval for Japanese queries
- Fully localized output for Japanese users# Oboyu (覚ゆ)


## Japanese Support

Oboyu provides exceptional support for Japanese language documents:

- Automatic encoding detection for Japanese text files (UTF-8, Shift-JIS, EUC-JP)
- Japanese-aware tokenization using MeCab/Sudachi
- Support for mixed Japanese-English content
- Japanese-optimized embedding models
- Accurate retrieval for Japanese queries

Example Japanese query:

```bash
oboyu query "システムの主要コンポーネントは何ですか？"
```
