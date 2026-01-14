#!/bin/bash
# Open Source Repository Ruleset Enforcer
# Enforces code quality rules locally without requiring GitHub API access

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
ENFORCE_STRICT=${ENFORCE_STRICT:-0}
ENFORCE_WARNINGS=${ENFORCE_WARNINGS:-1}

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((WARNINGS++))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Rule 1: Check file sizes
check_file_sizes() {
    log_info "Checking for large files..."
    local max_size=$((10 * 1024 * 1024))  # 10MB in bytes
    local large_files=0
    
    while IFS= read -r file; do
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt "$max_size" ]; then
            log_warning "Large file: $file ($(( size / 1024 / 1024 ))MB)"
            ((large_files++))
        fi
    done < <(git ls-files --exclude-standard)
    
    if [ "$large_files" -gt 0 ]; then
        log_warning "Found $large_files files larger than 10MB"
    else
        log_success "No excessively large files found"
    fi
}

# Rule 2: Check for debug statements
check_debug_statements() {
    log_info "Checking for debug statements..."
    local debug_count=0
    
    while IFS= read -r file; do
        if grep -q "print(" "$file" 2>/dev/null; then
            local count=$(grep -c "print(" "$file" || echo 0)
            if [ "$count" -gt 0 ]; then
                ((debug_count+=$count))
                log_warning "Print statements found in $file: $count"
            fi
        fi
    done < <(git ls-files "*.py")
    
    if [ "$debug_count" -gt 0 ]; then
        log_warning "Found $debug_count debug print statements"
        if [ "$ENFORCE_STRICT" -eq 1 ]; then
            log_error "Debug statements not allowed in strict mode"
        fi
    else
        log_success "No debug print statements found"
    fi
}

# Rule 3: Check for TODO/FIXME comments
check_code_comments() {
    log_info "Checking for unresolved TODOs and FIXMEs..."
    local todo_count=0
    
    while IFS= read -r file; do
        if grep -q "TODO\|FIXME" "$file" 2>/dev/null; then
            local count=$(grep -c "TODO\|FIXME" "$file" || echo 0)
            if [ "$count" -gt 0 ]; then
                ((todo_count+=$count))
                log_warning "$file has $count unresolved comments"
            fi
        fi
    done < <(git ls-files "*.py")
    
    if [ "$todo_count" -gt 0 ]; then
        log_warning "Found $todo_count unresolved TODO/FIXME comments"
    else
        log_success "No unresolved comments found"
    fi
}

# Rule 4: Check Python syntax
check_python_syntax() {
    log_info "Checking Python syntax..."
    local syntax_errors=0
    
    while IFS= read -r file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            log_error "Syntax error in $file"
            ((syntax_errors++))
        fi
    done < <(git ls-files "*.py")
    
    if [ "$syntax_errors" -eq 0 ]; then
        log_success "All Python files have valid syntax"
    else
        log_error "Found $syntax_errors Python syntax errors"
    fi
}

# Rule 5: Check for trailing whitespace
check_trailing_whitespace() {
    log_info "Checking for trailing whitespace..."
    local files_with_whitespace=0
    
    while IFS= read -r file; do
        if grep -q '[[:space:]]$' "$file" 2>/dev/null; then
            log_warning "Trailing whitespace in $file"
            ((files_with_whitespace++))
        fi
    done < <(git ls-files)
    
    if [ "$files_with_whitespace" -eq 0 ]; then
        log_success "No trailing whitespace found"
    else
        log_warning "Found $files_with_whitespace files with trailing whitespace"
    fi
}

# Rule 6: Check for proper .gitignore
check_gitignore() {
    log_info "Checking .gitignore configuration..."
    
    if [ ! -f "$REPO_ROOT/.gitignore" ]; then
        log_error ".gitignore file not found"
    elif [ ! -s "$REPO_ROOT/.gitignore" ]; then
        log_warning ".gitignore is empty"
    else
        # Check for common patterns
        if grep -q "__pycache__" "$REPO_ROOT/.gitignore"; then
            log_success ".gitignore includes __pycache__"
        else
            log_warning ".gitignore missing __pycache__ pattern"
        fi
        
        if grep -q "\.pyc" "$REPO_ROOT/.gitignore"; then
            log_success ".gitignore includes *.pyc"
        else
            log_warning ".gitignore missing *.pyc pattern"
        fi
    fi
}

# Rule 7: Check branch naming conventions
check_branch_conventions() {
    log_info "Checking branch naming conventions..."
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    # Allowed patterns: main, develop, feature/*, bugfix/*, hotfix/*, release/*
    if [[ "$current_branch" =~ ^(main|develop|feature|bugfix|hotfix|release)(/|$) ]]; then
        log_success "Branch name follows conventions: $current_branch"
    else
        log_warning "Non-standard branch name: $current_branch (consider using feature/*, bugfix/*, etc.)"
    fi
}

# Rule 8: Check commit messages (when applicable)
check_commit_conventions() {
    log_info "Checking recent commit message format..."
    
    # Get the last commit message
    local last_commit=$(git log -1 --pretty=%B 2>/dev/null || echo "")
    
    if [ -z "$last_commit" ]; then
        log_warning "No commits found yet"
    elif echo "$last_commit" | grep -qE '^(feat|fix|docs|style|refactor|test|chore):'; then
        log_success "Recent commits follow conventional format"
    else
        log_warning "Commits should follow conventional format: feat|fix|docs|style|refactor|test|chore: message"
    fi
}

# Main execution
main() {
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Open Source Repository Ruleset Enforcer${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
    
    check_file_sizes
    check_debug_statements
    check_code_comments
    check_python_syntax
    check_trailing_whitespace
    check_gitignore
    check_branch_conventions
    check_commit_conventions
    
    echo -e "\n${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Ruleset Enforcement Summary${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
    
    if [ "$ERRORS" -gt 0 ]; then
        echo -e "${RED}Errors: $ERRORS${NC}"
    else
        echo -e "${GREEN}Errors: 0${NC}"
    fi
    
    if [ "$WARNINGS" -gt 0 ]; then
        echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    else
        echo -e "${GREEN}Warnings: 0${NC}"
    fi
    
    echo ""
    
    if [ "$ERRORS" -gt 0 ] && [ "$ENFORCE_STRICT" -eq 1 ]; then
        log_error "Ruleset enforcement failed (strict mode)"
        return 1
    elif [ "$ERRORS" -gt 0 ]; then
        log_warning "Ruleset enforcement passed with errors (non-strict mode)"
        return 0
    else
        log_success "All rulesets passed!"
        return 0
    fi
}

main "$@"
