"""Tests for language-aware comment syntax generation."""

import pytest

from lask_lm.models import (
    LaskPrompt,
    LaskDirective,
    Contract,
    get_comment_syntax,
    LANGUAGE_COMMENT_SYNTAX,
    EXTENSION_TO_LANGUAGE,
)


class TestGetCommentSyntax:
    """Tests for the get_comment_syntax utility function."""

    # =========================================================================
    # Tests for explicit language parameter
    # =========================================================================

    def test_python_language(self):
        """Python language returns hash comment."""
        prefix, suffix = get_comment_syntax(language="python")
        assert prefix == "#"
        assert suffix == ""

    def test_csharp_language(self):
        """C# language returns C-style comment."""
        prefix, suffix = get_comment_syntax(language="csharp")
        assert prefix == "//"
        assert suffix == ""

    def test_javascript_language(self):
        """JavaScript language returns C-style comment."""
        prefix, suffix = get_comment_syntax(language="javascript")
        assert prefix == "//"
        assert suffix == ""

    def test_typescript_language(self):
        """TypeScript language returns C-style comment."""
        prefix, suffix = get_comment_syntax(language="typescript")
        assert prefix == "//"
        assert suffix == ""

    def test_html_language(self):
        """HTML language returns XML-style comment with suffix."""
        prefix, suffix = get_comment_syntax(language="html")
        assert prefix == "<!--"
        assert suffix == "-->"

    def test_css_language(self):
        """CSS language returns block comment with suffix."""
        prefix, suffix = get_comment_syntax(language="css")
        assert prefix == "/*"
        assert suffix == "*/"

    def test_sql_language(self):
        """SQL language returns double-dash comment."""
        prefix, suffix = get_comment_syntax(language="sql")
        assert prefix == "--"
        assert suffix == ""

    def test_ruby_language(self):
        """Ruby language returns hash comment."""
        prefix, suffix = get_comment_syntax(language="ruby")
        assert prefix == "#"
        assert suffix == ""

    def test_language_case_insensitive(self):
        """Language matching is case-insensitive."""
        assert get_comment_syntax(language="PYTHON") == ("#", "")
        assert get_comment_syntax(language="Python") == ("#", "")
        assert get_comment_syntax(language="CSharp") == ("//", "")
        assert get_comment_syntax(language="HTML") == ("<!--", "-->")

    def test_unknown_language_defaults_to_c_style(self):
        """Unknown language defaults to C-style comments."""
        prefix, suffix = get_comment_syntax(language="unknown_language")
        assert prefix == "//"
        assert suffix == ""

    # =========================================================================
    # Tests for file extension inference
    # =========================================================================

    def test_python_extension(self):
        """Python file extension returns hash comment."""
        assert get_comment_syntax(file_path="script.py") == ("#", "")
        assert get_comment_syntax(file_path="module.pyw") == ("#", "")
        assert get_comment_syntax(file_path="types.pyi") == ("#", "")

    def test_csharp_extension(self):
        """C# file extension returns C-style comment."""
        assert get_comment_syntax(file_path="UserService.cs") == ("//", "")

    def test_javascript_extensions(self):
        """JavaScript file extensions return C-style comment."""
        assert get_comment_syntax(file_path="app.js") == ("//", "")
        assert get_comment_syntax(file_path="module.mjs") == ("//", "")
        assert get_comment_syntax(file_path="common.cjs") == ("//", "")
        assert get_comment_syntax(file_path="Component.jsx") == ("//", "")

    def test_typescript_extensions(self):
        """TypeScript file extensions return C-style comment."""
        assert get_comment_syntax(file_path="app.ts") == ("//", "")
        assert get_comment_syntax(file_path="Component.tsx") == ("//", "")
        assert get_comment_syntax(file_path="module.mts") == ("//", "")
        assert get_comment_syntax(file_path="common.cts") == ("//", "")

    def test_java_extension(self):
        """Java file extension returns C-style comment."""
        assert get_comment_syntax(file_path="Main.java") == ("//", "")

    def test_c_cpp_extensions(self):
        """C/C++ file extensions return C-style comment."""
        assert get_comment_syntax(file_path="main.c") == ("//", "")
        assert get_comment_syntax(file_path="header.h") == ("//", "")
        assert get_comment_syntax(file_path="main.cpp") == ("//", "")
        assert get_comment_syntax(file_path="main.cc") == ("//", "")
        assert get_comment_syntax(file_path="header.hpp") == ("//", "")

    def test_go_extension(self):
        """Go file extension returns C-style comment."""
        assert get_comment_syntax(file_path="main.go") == ("//", "")

    def test_rust_extension(self):
        """Rust file extension returns C-style comment."""
        assert get_comment_syntax(file_path="main.rs") == ("//", "")

    def test_html_extensions(self):
        """HTML file extensions return XML-style comment."""
        assert get_comment_syntax(file_path="index.html") == ("<!--", "-->")
        assert get_comment_syntax(file_path="page.htm") == ("<!--", "-->")
        assert get_comment_syntax(file_path="doc.xhtml") == ("<!--", "-->")

    def test_xml_extensions(self):
        """XML file extensions return XML-style comment."""
        assert get_comment_syntax(file_path="config.xml") == ("<!--", "-->")
        assert get_comment_syntax(file_path="style.xsl") == ("<!--", "-->")
        assert get_comment_syntax(file_path="transform.xslt") == ("<!--", "-->")

    def test_css_extension(self):
        """CSS file extension returns block comment."""
        assert get_comment_syntax(file_path="styles.css") == ("/*", "*/")

    def test_scss_sass_less_extensions(self):
        """SCSS/Sass/Less file extensions return C-style comment."""
        assert get_comment_syntax(file_path="styles.scss") == ("//", "")
        assert get_comment_syntax(file_path="styles.sass") == ("//", "")
        assert get_comment_syntax(file_path="styles.less") == ("//", "")

    def test_sql_extension(self):
        """SQL file extension returns double-dash comment."""
        assert get_comment_syntax(file_path="query.sql") == ("--", "")

    def test_yaml_extensions(self):
        """YAML file extensions return hash comment."""
        assert get_comment_syntax(file_path="config.yaml") == ("#", "")
        assert get_comment_syntax(file_path="config.yml") == ("#", "")

    def test_shell_extensions(self):
        """Shell file extensions return hash comment."""
        assert get_comment_syntax(file_path="script.sh") == ("#", "")
        assert get_comment_syntax(file_path="script.bash") == ("#", "")
        assert get_comment_syntax(file_path="script.zsh") == ("#", "")

    def test_file_path_with_directory(self):
        """File path with directory still infers correctly."""
        assert get_comment_syntax(file_path="src/components/App.tsx") == ("//", "")
        assert get_comment_syntax(file_path="/home/user/scripts/run.py") == ("#", "")
        assert get_comment_syntax(file_path="./templates/index.html") == ("<!--", "-->")

    def test_file_path_case_insensitive(self):
        """File extension matching is case-insensitive."""
        assert get_comment_syntax(file_path="Script.PY") == ("#", "")
        assert get_comment_syntax(file_path="PAGE.HTML") == ("<!--", "-->")
        assert get_comment_syntax(file_path="Style.CSS") == ("/*", "*/")

    def test_unknown_extension_defaults_to_c_style(self):
        """Unknown file extension defaults to C-style comments."""
        prefix, suffix = get_comment_syntax(file_path="file.unknown")
        assert prefix == "//"
        assert suffix == ""

    # =========================================================================
    # Tests for priority order
    # =========================================================================

    def test_language_takes_precedence_over_extension(self):
        """Explicit language parameter takes precedence over file extension."""
        # File is .py but language is html
        prefix, suffix = get_comment_syntax(file_path="script.py", language="html")
        assert prefix == "<!--"
        assert suffix == "-->"

    def test_no_arguments_defaults_to_c_style(self):
        """No arguments defaults to C-style comments."""
        prefix, suffix = get_comment_syntax()
        assert prefix == "//"
        assert suffix == ""

    def test_none_arguments_defaults_to_c_style(self):
        """None arguments defaults to C-style comments."""
        prefix, suffix = get_comment_syntax(file_path=None, language=None)
        assert prefix == "//"
        assert suffix == ""


class TestLaskPromptToCommentLanguageAware:
    """Tests for LaskPrompt.to_comment() with language-aware syntax."""

    # =========================================================================
    # Tests for file extension inference
    # =========================================================================

    def test_python_file_uses_hash_comment(self):
        """Python file gets hash comment syntax."""
        prompt = LaskPrompt(file_path="script.py", intent="Add logging")
        comment = prompt.to_comment()
        assert comment == "# @ Add logging"

    def test_csharp_file_uses_c_style_comment(self):
        """C# file gets C-style comment syntax."""
        prompt = LaskPrompt(file_path="UserService.cs", intent="Create service")
        comment = prompt.to_comment()
        assert comment == "// @ Create service"

    def test_javascript_file_uses_c_style_comment(self):
        """JavaScript file gets C-style comment syntax."""
        prompt = LaskPrompt(file_path="app.js", intent="Initialize app")
        comment = prompt.to_comment()
        assert comment == "// @ Initialize app"

    def test_typescript_file_uses_c_style_comment(self):
        """TypeScript file gets C-style comment syntax."""
        prompt = LaskPrompt(file_path="Component.tsx", intent="Create component")
        comment = prompt.to_comment()
        assert comment == "// @ Create component"

    def test_html_file_uses_xml_comment(self):
        """HTML file gets XML-style comment with closing tag."""
        prompt = LaskPrompt(file_path="index.html", intent="Add header")
        comment = prompt.to_comment()
        assert comment == "<!-- @ Add header -->"

    def test_css_file_uses_block_comment(self):
        """CSS file gets block comment with closing tag."""
        prompt = LaskPrompt(file_path="styles.css", intent="Add button styles")
        comment = prompt.to_comment()
        assert comment == "/* @ Add button styles */"

    def test_sql_file_uses_dash_comment(self):
        """SQL file gets double-dash comment syntax."""
        prompt = LaskPrompt(file_path="query.sql", intent="Select users")
        comment = prompt.to_comment()
        assert comment == "-- @ Select users"

    def test_yaml_file_uses_hash_comment(self):
        """YAML file gets hash comment syntax."""
        prompt = LaskPrompt(file_path="config.yaml", intent="Add settings")
        comment = prompt.to_comment()
        assert comment == "# @ Add settings"

    def test_ruby_file_uses_hash_comment(self):
        """Ruby file gets hash comment syntax."""
        prompt = LaskPrompt(file_path="script.rb", intent="Define class")
        comment = prompt.to_comment()
        assert comment == "# @ Define class"

    # =========================================================================
    # Tests for explicit language parameter
    # =========================================================================

    def test_explicit_language_overrides_extension(self):
        """Explicit language parameter overrides file extension inference."""
        # File is .txt but we specify python
        prompt = LaskPrompt(file_path="code.txt", intent="Add function")
        comment = prompt.to_comment(language="python")
        assert comment == "# @ Add function"

    def test_html_language_parameter(self):
        """HTML language parameter produces correct syntax."""
        prompt = LaskPrompt(file_path="template.custom", intent="Add div")
        comment = prompt.to_comment(language="html")
        assert comment == "<!-- @ Add div -->"

    # =========================================================================
    # Tests for explicit comment_prefix (backward compatibility)
    # =========================================================================

    def test_explicit_prefix_overrides_all(self):
        """Explicit comment_prefix takes precedence over language detection."""
        prompt = LaskPrompt(file_path="script.py", intent="Add logging")
        comment = prompt.to_comment(comment_prefix="//")
        assert comment == "// @ Add logging"

    def test_explicit_prefix_ignores_language(self):
        """Explicit comment_prefix ignores language parameter."""
        prompt = LaskPrompt(file_path="script.py", intent="Add logging")
        comment = prompt.to_comment(comment_prefix="--", language="html")
        assert comment == "-- @ Add logging"

    # =========================================================================
    # Tests for directives with language-aware syntax
    # =========================================================================

    def test_python_file_with_directives(self):
        """Python file with directives uses correct syntax."""
        prompt = LaskPrompt(
            file_path="script.py",
            intent="Validate input",
            directives=[
                LaskDirective(directive_type="context", value="utils.py"),
                LaskDirective(directive_type="model", value="gpt-4"),
            ],
        )
        comment = prompt.to_comment()
        assert comment == "# @ @context(utils.py) @model(gpt-4) Validate input"

    def test_html_file_with_directives(self):
        """HTML file with directives uses correct syntax with closing tag."""
        prompt = LaskPrompt(
            file_path="page.html",
            intent="Create form",
            directives=[
                LaskDirective(directive_type="context", value="template.html"),
            ],
        )
        comment = prompt.to_comment()
        assert comment == "<!-- @ @context(template.html) Create form -->"

    def test_css_file_with_directives(self):
        """CSS file with directives uses correct syntax with closing tag."""
        prompt = LaskPrompt(
            file_path="styles.css",
            intent="Add responsive styles",
            directives=[
                LaskDirective(directive_type="context", value="base.css"),
            ],
        )
        comment = prompt.to_comment()
        assert comment == "/* @ @context(base.css) Add responsive styles */"

    # =========================================================================
    # Tests for contracts with language-aware syntax
    # =========================================================================

    def test_python_file_with_contracts(self):
        """Python file with contracts uses correct syntax."""
        prompt = LaskPrompt(
            file_path="service.py",
            intent="Create user service",
            resolved_contracts=[
                Contract(
                    name="IUserRepository.get",
                    signature="User get(id: int)",
                    description="Gets user by ID",
                    context_files=["repository.py"],
                ),
            ],
        )
        comment = prompt.to_comment()
        assert comment.startswith("# @")
        assert "@context(repository.py)" in comment
        assert "[requires: IUserRepository.get: User get(id: int)]" in comment

    def test_html_file_with_contracts(self):
        """HTML file with contracts uses correct syntax with closing tag."""
        prompt = LaskPrompt(
            file_path="component.html",
            intent="Render user list",
            resolved_contracts=[
                Contract(
                    name="UserData",
                    signature="{ users: User[] }",
                    description="User data",
                ),
            ],
        )
        comment = prompt.to_comment()
        assert comment.startswith("<!--")
        assert comment.endswith("-->")
        assert "[requires: UserData: { users: User[] }]" in comment

    # =========================================================================
    # Tests for DELETE operations with language-aware syntax
    # =========================================================================

    def test_python_delete_operation(self):
        """Python file DELETE operation uses correct syntax."""
        prompt = LaskPrompt(
            file_path="old_code.py",
            intent="Remove deprecated function",
            replaces="legacy_function",
            is_delete=True,
        )
        comment = prompt.to_comment()
        assert comment == "# @delete legacy_function"

    def test_html_delete_operation(self):
        """HTML file DELETE operation uses correct syntax with closing tag."""
        prompt = LaskPrompt(
            file_path="page.html",
            intent="Remove old section",
            replaces="deprecated div",
            is_delete=True,
        )
        comment = prompt.to_comment()
        assert comment == "<!-- @delete deprecated div -->"

    def test_css_delete_operation(self):
        """CSS file DELETE operation uses correct syntax with closing tag."""
        prompt = LaskPrompt(
            file_path="styles.css",
            intent="Remove old styles",
            replaces="legacy-class",
            is_delete=True,
        )
        comment = prompt.to_comment()
        assert comment == "/* @delete legacy-class */"

    def test_delete_without_replaces_python(self):
        """Python DELETE without replaces uses default text."""
        prompt = LaskPrompt(
            file_path="code.py",
            intent="Remove code",
            is_delete=True,
        )
        comment = prompt.to_comment()
        assert comment == "# @delete target code"

    def test_delete_without_replaces_html(self):
        """HTML DELETE without replaces uses default text with closing tag."""
        prompt = LaskPrompt(
            file_path="page.html",
            intent="Remove element",
            is_delete=True,
        )
        comment = prompt.to_comment()
        assert comment == "<!-- @delete target code -->"


class TestLanguageMappingCompleteness:
    """Tests to verify the language and extension mappings are complete."""

    def test_all_languages_have_valid_syntax(self):
        """All languages in LANGUAGE_COMMENT_SYNTAX have valid (prefix, suffix) tuples."""
        for lang, syntax in LANGUAGE_COMMENT_SYNTAX.items():
            assert isinstance(syntax, tuple), f"Language {lang} syntax should be tuple"
            assert len(syntax) == 2, f"Language {lang} syntax should have 2 elements"
            prefix, suffix = syntax
            assert isinstance(prefix, str), f"Language {lang} prefix should be string"
            assert isinstance(suffix, str), f"Language {lang} suffix should be string"
            assert len(prefix) > 0, f"Language {lang} prefix should not be empty"

    def test_all_extensions_map_to_known_languages(self):
        """All extensions in EXTENSION_TO_LANGUAGE map to known languages."""
        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            assert lang in LANGUAGE_COMMENT_SYNTAX, (
                f"Extension {ext} maps to unknown language {lang}"
            )

    def test_common_languages_covered(self):
        """Common programming languages are covered."""
        common_languages = [
            "python", "csharp", "javascript", "typescript", "java",
            "c", "cpp", "go", "rust", "ruby", "html", "css", "sql",
        ]
        for lang in common_languages:
            assert lang in LANGUAGE_COMMENT_SYNTAX, f"Language {lang} should be covered"

    def test_common_extensions_covered(self):
        """Common file extensions are covered."""
        common_extensions = [
            ".py", ".cs", ".js", ".ts", ".java", ".c", ".cpp", ".go",
            ".rs", ".rb", ".html", ".css", ".sql", ".yaml", ".json",
        ]
        for ext in common_extensions:
            # .json doesn't have comments so it's okay if it's missing
            if ext != ".json":
                assert ext in EXTENSION_TO_LANGUAGE, f"Extension {ext} should be covered"
