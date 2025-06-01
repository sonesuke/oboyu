# VHS configuration
Output demo.gif
Set FontSize 18
Set Width 1200
Set Height 800
Set Theme "Dracula"

# Show title
Type "# Oboyu - Self-hosted MCP Japanese search system" Enter Enter
Sleep 2s

# Index documents
Type "# Index your documents" Enter
Type "oboyu index ~/oboyu/main/docs" Enter
Sleep 15s
Enter

# Search documents
Type "# Search for information" Enter
Type "oboyu query --query 'What is Oboyu?'" Enter
Sleep 6
Enter

# Interactive mode
Type "# Or use interactive mode" Enter
Type "oboyu query --interactive" Enter
Sleep 5s
Type "What is Oboyu?" Enter
Sleep 3s
Type "/exit" Enter
Sleep 1s

# End
Type "# That's it! Simple and powerful document search 🚀" Enter
Sleep 3s