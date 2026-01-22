#!/usr/bin/env python3
"""
DSL Compiler for VLIW SIMD Architecture

Compiles a high-level Python-like DSL into optimized VLIW instruction streams.

Features:
- Python-like syntax for algorithm expression
- Automatic vectorization (VLEN=8)
- Automatic dependency analysis and scheduling
- VLIW packing for parallel execution
- KernelBuilder-compatible output

Usage:
    # Parse and compile a DSL file
    python tools/dsl_compiler/dsl_compiler.py example.dsl

    # Output JSON
    python tools/dsl_compiler/dsl_compiler.py example.dsl --json

    # From Python
    from tools.dsl_compiler.dsl_compiler import DSLCompiler
    compiler = DSLCompiler()
    result = compiler.compile(dsl_source)
"""

import sys
import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
from enum import Enum, auto
from copy import deepcopy

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, HASH_STAGES

# Try to import Rich for better formatting (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============== Data Types ==============

class OpType(Enum):
    """Operation types for IR."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    XOR = auto()
    AND = auto()
    OR = auto()
    SHL = auto()
    SHR = auto()
    LT = auto()
    EQ = auto()
    CONST = auto()
    LOAD = auto()
    STORE = auto()
    VLOAD = auto()
    VSTORE = auto()
    VBROADCAST = auto()
    SELECT = auto()
    VSELECT = auto()
    MULTIPLY_ADD = auto()
    HASH = auto()  # High-level hash operation


# Map DSL operators to IR ops
OP_MAP = {
    '+': OpType.ADD,
    '-': OpType.SUB,
    '*': OpType.MUL,
    '//': OpType.DIV,
    '%': OpType.MOD,
    '^': OpType.XOR,
    '&': OpType.AND,
    '|': OpType.OR,
    '<<': OpType.SHL,
    '>>': OpType.SHR,
    '<': OpType.LT,
    '==': OpType.EQ,
}

# Map IR ops to VLIW operation strings
IR_TO_VLIW = {
    OpType.ADD: '+',
    OpType.SUB: '-',
    OpType.MUL: '*',
    OpType.DIV: '//',
    OpType.MOD: '%',
    OpType.XOR: '^',
    OpType.AND: '&',
    OpType.OR: '|',
    OpType.SHL: '<<',
    OpType.SHR: '>>',
    OpType.LT: '<',
    OpType.EQ: '==',
}


class ValueType(Enum):
    """Type of a value."""
    SCALAR = auto()
    VECTOR = auto()  # VLEN elements
    UNKNOWN = auto()


@dataclass
class IRValue:
    """A value in the IR."""
    id: int
    name: str
    vtype: ValueType
    is_constant: bool = False
    constant_value: Optional[int] = None


@dataclass
class IRNode:
    """A node in the IR DAG."""
    id: int
    op: OpType
    dest: IRValue
    sources: List[IRValue]
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Scheduling info
    scheduled_cycle: int = -1
    engine: str = ""


@dataclass
class IRLoop:
    """A loop in the IR."""
    id: int
    iterator: str
    start: int
    end: int
    step: int
    body: List['IRNode']


@dataclass
class IRBranch:
    """A conditional branch in the IR."""
    id: int
    condition: IRValue
    true_body: List[IRNode]
    false_body: List[IRNode]


@dataclass
class CompileResult:
    """Result of compilation."""
    success: bool
    instructions: List[dict]
    cycles: int
    scratch_used: int
    scratch_map: Dict[str, int]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "cycles": self.cycles,
            "scratch_used": self.scratch_used,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============== Lexer ==============

class TokenType(Enum):
    """Token types for the DSL lexer."""
    # Literals
    NUMBER = auto()
    IDENTIFIER = auto()
    STRING = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    CARET = auto()
    AMPERSAND = auto()
    PIPE = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    NE = auto()
    LE = auto()
    GE = auto()
    EQEQ = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    AT = auto()

    # Keywords
    FOR = auto()
    IN = auto()
    RANGE = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    DEF = auto()
    RETURN = auto()
    VAR = auto()
    VEC = auto()
    HASH = auto()
    LOAD = auto()
    STORE = auto()
    VLOAD = auto()
    VSTORE = auto()
    BROADCAST = auto()

    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass
class Token:
    """A lexer token."""
    type: TokenType
    value: Any
    line: int
    column: int


class Lexer:
    """Tokenize DSL source code."""

    KEYWORDS = {
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'range': TokenType.RANGE,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'elif': TokenType.ELIF,
        'def': TokenType.DEF,
        'return': TokenType.RETURN,
        'var': TokenType.VAR,
        'vec': TokenType.VEC,
        'hash': TokenType.HASH,
        'load': TokenType.LOAD,
        'store': TokenType.STORE,
        'vload': TokenType.VLOAD,
        'vstore': TokenType.VSTORE,
        'broadcast': TokenType.BROADCAST,
    }

    OPERATORS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.STAR,
        '/': TokenType.SLASH,
        '%': TokenType.PERCENT,
        '^': TokenType.CARET,
        '&': TokenType.AMPERSAND,
        '|': TokenType.PIPE,
        '<': TokenType.LT,
        '>': TokenType.GT,
        '=': TokenType.EQ,
        '<<': TokenType.LSHIFT,
        '>>': TokenType.RSHIFT,
        '==': TokenType.EQEQ,
        '!=': TokenType.NE,
        '<=': TokenType.LE,
        '>=': TokenType.GE,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.indent_stack = [0]

    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos >= len(self.source):
            return ''
        return self.source[pos]

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def skip_whitespace(self) -> bool:
        """Skip whitespace but not newlines. Returns True if any skipped."""
        skipped = False
        while self.peek() in ' \t':
            self.advance()
            skipped = True
        return skipped

    def skip_comment(self):
        """Skip # comments until end of line."""
        if self.peek() == '#':
            while self.peek() and self.peek() != '\n':
                self.advance()

    def read_number(self) -> Token:
        """Read an integer or hex literal."""
        start = self.pos
        line, col = self.line, self.column

        if self.peek() == '0' and self.peek(1) in 'xX':
            # Hex literal
            self.advance()  # 0
            self.advance()  # x
            while self.peek() and self.peek() in '0123456789abcdefABCDEF':
                self.advance()
            value = int(self.source[start:self.pos], 16)
        else:
            while self.peek().isdigit():
                self.advance()
            value = int(self.source[start:self.pos])

        return Token(TokenType.NUMBER, value, line, col)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start = self.pos
        line, col = self.line, self.column

        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            self.advance()

        text = self.source[start:self.pos]
        ttype = self.KEYWORDS.get(text, TokenType.IDENTIFIER)
        return Token(ttype, text, line, col)

    def read_operator(self) -> Token:
        """Read an operator (possibly multi-character)."""
        line, col = self.line, self.column

        # Try two-char operators first
        two_char = self.peek() + self.peek(1)
        if two_char in self.OPERATORS:
            self.advance()
            self.advance()
            return Token(self.OPERATORS[two_char], two_char, line, col)

        # Single-char operators
        ch = self.advance()
        if ch in self.OPERATORS:
            return Token(self.OPERATORS[ch], ch, line, col)

        # Special delimiters
        delimiters = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
            '.': TokenType.DOT,
            '@': TokenType.AT,
        }
        if ch in delimiters:
            return Token(delimiters[ch], ch, line, col)

        raise SyntaxError(f"Unknown character '{ch}' at line {line}, column {col}")

    def handle_indentation(self):
        """Handle Python-style indentation at start of line."""
        line, col = self.line, self.column

        # Count leading spaces
        indent = 0
        while self.peek() == ' ':
            self.advance()
            indent += 1
        while self.peek() == '\t':
            self.advance()
            indent += 4  # Treat tabs as 4 spaces

        # Skip blank lines and comments
        if self.peek() == '\n' or self.peek() == '#':
            return

        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            self.indent_stack.append(indent)
            self.tokens.append(Token(TokenType.INDENT, indent, line, col))
        elif indent < current_indent:
            while self.indent_stack and indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, indent, line, col))

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source."""
        at_line_start = True

        while self.pos < len(self.source):
            # Handle indentation at line start
            if at_line_start:
                self.handle_indentation()
                at_line_start = False

            ch = self.peek()

            # Skip whitespace (but not newlines)
            if ch in ' \t':
                self.skip_whitespace()
                continue

            # Skip comments
            if ch == '#':
                self.skip_comment()
                continue

            # Newline
            if ch == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                at_line_start = True
                continue

            # Number
            if ch.isdigit():
                self.tokens.append(self.read_number())
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == '_':
                self.tokens.append(self.read_identifier())
                continue

            # Operators and delimiters
            self.tokens.append(self.read_operator())

        # Emit remaining dedents
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, 0, self.line, self.column))

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


# ============== Parser ==============

@dataclass
class ASTNode:
    """Base AST node."""
    pass


@dataclass
class ASTNumber(ASTNode):
    value: int


@dataclass
class ASTIdentifier(ASTNode):
    name: str


@dataclass
class ASTBinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class ASTUnaryOp(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class ASTCall(ASTNode):
    name: str
    args: List[ASTNode]


@dataclass
class ASTIndex(ASTNode):
    base: ASTNode
    index: ASTNode


@dataclass
class ASTAssign(ASTNode):
    target: ASTNode
    value: ASTNode


@dataclass
class ASTFor(ASTNode):
    iterator: str
    start: ASTNode
    end: ASTNode
    step: ASTNode
    body: List[ASTNode]


@dataclass
class ASTIf(ASTNode):
    condition: ASTNode
    true_body: List[ASTNode]
    false_body: List[ASTNode]


@dataclass
class ASTFunction(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]


@dataclass
class ASTDecorator(ASTNode):
    name: str
    args: List[ASTNode]
    target: ASTNode


@dataclass
class ASTVarDecl(ASTNode):
    name: str
    vtype: str  # 'scalar' or 'vector'
    size: Optional[int]  # For vectors


class Parser:
    """Parse DSL tokens into an AST."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, ttype: TokenType) -> Token:
        tok = self.advance()
        if tok.type != ttype:
            raise SyntaxError(f"Expected {ttype.name}, got {tok.type.name} at line {tok.line}")
        return tok

    def skip_newlines(self):
        while self.peek().type == TokenType.NEWLINE:
            self.advance()

    def parse(self) -> List[ASTNode]:
        """Parse the entire program."""
        statements = []
        self.skip_newlines()

        while self.peek().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        return statements

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        tok = self.peek()

        if tok.type == TokenType.AT:
            return self.parse_decorator()
        elif tok.type == TokenType.DEF:
            return self.parse_function()
        elif tok.type == TokenType.FOR:
            return self.parse_for()
        elif tok.type == TokenType.IF:
            return self.parse_if()
        elif tok.type == TokenType.VAR:
            return self.parse_var_decl()
        elif tok.type == TokenType.VEC:
            return self.parse_vec_decl()
        elif tok.type == TokenType.IDENTIFIER:
            return self.parse_assignment_or_expr()
        elif tok.type in (TokenType.STORE, TokenType.VSTORE, TokenType.HASH,
                          TokenType.LOAD, TokenType.VLOAD, TokenType.BROADCAST):
            # Builtin function call as statement
            return self.parse_builtin()
        elif tok.type == TokenType.NEWLINE:
            self.advance()
            return None
        else:
            raise SyntaxError(f"Unexpected token {tok.type.name} at line {tok.line}")

    def parse_decorator(self) -> ASTDecorator:
        """Parse @decorator."""
        self.expect(TokenType.AT)
        name = self.expect(TokenType.IDENTIFIER).value

        args = []
        if self.peek().type == TokenType.LPAREN:
            self.advance()
            if self.peek().type != TokenType.RPAREN:
                args.append(self.parse_expression())
                while self.peek().type == TokenType.COMMA:
                    self.advance()
                    args.append(self.parse_expression())
            self.expect(TokenType.RPAREN)

        self.skip_newlines()
        target = self.parse_statement()

        return ASTDecorator(name, args, target)

    def parse_function(self) -> ASTFunction:
        """Parse function definition."""
        self.expect(TokenType.DEF)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)

        params = []
        if self.peek().type == TokenType.IDENTIFIER:
            params.append(self.advance().value)
            while self.peek().type == TokenType.COMMA:
                self.advance()
                params.append(self.expect(TokenType.IDENTIFIER).value)

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        self.skip_newlines()

        body = self.parse_block()
        return ASTFunction(name, params, body)

    def parse_for(self) -> ASTFor:
        """Parse for loop."""
        self.expect(TokenType.FOR)
        iterator = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.IN)
        self.expect(TokenType.RANGE)
        self.expect(TokenType.LPAREN)

        # Parse range(start, end) or range(end) or range(start, end, step)
        args = [self.parse_expression()]
        while self.peek().type == TokenType.COMMA:
            self.advance()
            args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        self.skip_newlines()

        body = self.parse_block()

        # Normalize range arguments
        if len(args) == 1:
            start, end, step = ASTNumber(0), args[0], ASTNumber(1)
        elif len(args) == 2:
            start, end, step = args[0], args[1], ASTNumber(1)
        else:
            start, end, step = args[0], args[1], args[2]

        return ASTFor(iterator, start, end, step, body)

    def parse_if(self) -> ASTIf:
        """Parse if statement."""
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()

        true_body = self.parse_block()
        false_body = []

        self.skip_newlines()
        if self.peek().type == TokenType.ELSE:
            self.advance()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            false_body = self.parse_block()
        elif self.peek().type == TokenType.ELIF:
            false_body = [self.parse_if()]

        return ASTIf(condition, true_body, false_body)

    def parse_var_decl(self) -> ASTVarDecl:
        """Parse scalar variable declaration."""
        self.expect(TokenType.VAR)
        name = self.expect(TokenType.IDENTIFIER).value
        return ASTVarDecl(name, 'scalar', None)

    def parse_vec_decl(self) -> ASTVarDecl:
        """Parse vector variable declaration."""
        self.expect(TokenType.VEC)
        name = self.expect(TokenType.IDENTIFIER).value

        size = VLEN
        if self.peek().type == TokenType.LBRACKET:
            self.advance()
            size = self.expect(TokenType.NUMBER).value
            self.expect(TokenType.RBRACKET)

        return ASTVarDecl(name, 'vector', size)

    def parse_block(self) -> List[ASTNode]:
        """Parse an indented block."""
        self.expect(TokenType.INDENT)
        statements = []

        while self.peek().type not in (TokenType.DEDENT, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        if self.peek().type == TokenType.DEDENT:
            self.advance()

        return statements

    def parse_assignment_or_expr(self) -> ASTNode:
        """Parse assignment or expression statement."""
        expr = self.parse_expression()

        if self.peek().type == TokenType.EQ:
            self.advance()
            value = self.parse_expression()
            return ASTAssign(expr, value)

        return expr

    def parse_expression(self) -> ASTNode:
        """Parse expression with operator precedence."""
        return self.parse_or()

    def parse_or(self) -> ASTNode:
        left = self.parse_xor()
        while self.peek().type == TokenType.PIPE:
            op = self.advance().value
            right = self.parse_xor()
            left = ASTBinOp(op, left, right)
        return left

    def parse_xor(self) -> ASTNode:
        left = self.parse_and()
        while self.peek().type == TokenType.CARET:
            op = self.advance().value
            right = self.parse_and()
            left = ASTBinOp(op, left, right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_comparison()
        while self.peek().type == TokenType.AMPERSAND:
            op = self.advance().value
            right = self.parse_comparison()
            left = ASTBinOp(op, left, right)
        return left

    def parse_comparison(self) -> ASTNode:
        left = self.parse_shift()
        while self.peek().type in (TokenType.LT, TokenType.GT, TokenType.LE,
                                    TokenType.GE, TokenType.EQEQ, TokenType.NE):
            op = self.advance().value
            right = self.parse_shift()
            left = ASTBinOp(op, left, right)
        return left

    def parse_shift(self) -> ASTNode:
        left = self.parse_additive()
        while self.peek().type in (TokenType.LSHIFT, TokenType.RSHIFT):
            op = self.advance().value
            right = self.parse_additive()
            left = ASTBinOp(op, left, right)
        return left

    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = ASTBinOp(op, left, right)
        return left

    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = ASTBinOp(op, left, right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.peek().type == TokenType.MINUS:
            op = self.advance().value
            operand = self.parse_unary()
            return ASTUnaryOp(op, operand)
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        expr = self.parse_primary()

        while True:
            if self.peek().type == TokenType.LBRACKET:
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = ASTIndex(expr, index)
            elif self.peek().type == TokenType.LPAREN and isinstance(expr, ASTIdentifier):
                self.advance()
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    while self.peek().type == TokenType.COMMA:
                        self.advance()
                        args.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                expr = ASTCall(expr.name, args)
            else:
                break

        return expr

    def parse_primary(self) -> ASTNode:
        """Parse primary expression."""
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            return ASTNumber(tok.value)
        elif tok.type == TokenType.IDENTIFIER:
            self.advance()
            return ASTIdentifier(tok.value)
        elif tok.type in (TokenType.HASH, TokenType.LOAD, TokenType.STORE,
                          TokenType.VLOAD, TokenType.VSTORE, TokenType.BROADCAST):
            return self.parse_builtin()
        elif tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        else:
            raise SyntaxError(f"Unexpected token {tok.type.name} at line {tok.line}")

    def parse_builtin(self) -> ASTCall:
        """Parse builtin function call."""
        tok = self.advance()
        self.expect(TokenType.LPAREN)

        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            while self.peek().type == TokenType.COMMA:
                self.advance()
                args.append(self.parse_expression())

        self.expect(TokenType.RPAREN)
        return ASTCall(tok.value, args)


# ============== IR Builder ==============

class IRBuilder:
    """Build IR from AST."""

    def __init__(self):
        self.nodes: List[IRNode] = []
        self.values: Dict[str, IRValue] = {}
        self.next_id = 0
        self.constants: Dict[int, IRValue] = {}
        self.vectorize_hint = False
        self.vectorize_size = VLEN

    def new_id(self) -> int:
        id = self.next_id
        self.next_id += 1
        return id

    def get_or_create_value(self, name: str, vtype: ValueType = ValueType.SCALAR) -> IRValue:
        """Get existing value or create new one."""
        if name not in self.values:
            self.values[name] = IRValue(self.new_id(), name, vtype)
        return self.values[name]

    def get_constant(self, value: int) -> IRValue:
        """Get or create constant value."""
        if value not in self.constants:
            v = IRValue(self.new_id(), f"const_{value}", ValueType.SCALAR, True, value)
            self.constants[value] = v
        return self.constants[value]

    def build(self, ast: List[ASTNode]) -> List[IRNode]:
        """Build IR from AST."""
        for node in ast:
            self.build_statement(node)
        return self.nodes

    def build_statement(self, node: ASTNode):
        """Build IR for a statement."""
        if isinstance(node, ASTAssign):
            self.build_assignment(node)
        elif isinstance(node, ASTFor):
            self.build_for(node)
        elif isinstance(node, ASTIf):
            self.build_if(node)
        elif isinstance(node, ASTVarDecl):
            self.build_var_decl(node)
        elif isinstance(node, ASTDecorator):
            self.build_decorator(node)
        elif isinstance(node, ASTFunction):
            self.build_function(node)
        elif isinstance(node, ASTCall):
            self.build_expression(node)

    def build_assignment(self, node: ASTAssign):
        """Build IR for assignment."""
        value_ir = self.build_expression(node.value)

        if isinstance(node.target, ASTIdentifier):
            dest = self.get_or_create_value(node.target.name, value_ir.vtype)
            # Copy/move the value
            ir_node = IRNode(self.new_id(), OpType.ADD, dest,
                           [value_ir, self.get_constant(0)])
            self.nodes.append(ir_node)
        elif isinstance(node.target, ASTIndex):
            # Store to memory
            base = self.build_expression(node.target.base)
            index = self.build_expression(node.target.index)
            addr = IRValue(self.new_id(), f"addr_{self.new_id()}", ValueType.SCALAR)

            # Compute address
            addr_node = IRNode(self.new_id(), OpType.ADD, addr, [base, index])
            self.nodes.append(addr_node)

            # Store
            store_node = IRNode(self.new_id(), OpType.STORE,
                              IRValue(self.new_id(), "_", ValueType.SCALAR),
                              [addr, value_ir])
            self.nodes.append(store_node)

    def build_expression(self, node: ASTNode) -> IRValue:
        """Build IR for expression, return result value."""
        if isinstance(node, ASTNumber):
            return self.get_constant(node.value)

        elif isinstance(node, ASTIdentifier):
            return self.get_or_create_value(node.name)

        elif isinstance(node, ASTBinOp):
            left = self.build_expression(node.left)
            right = self.build_expression(node.right)

            # Determine result type
            result_type = ValueType.VECTOR if (left.vtype == ValueType.VECTOR or
                                               right.vtype == ValueType.VECTOR) else ValueType.SCALAR

            op = OP_MAP.get(node.op)
            if not op:
                raise ValueError(f"Unknown operator: {node.op}")

            result = IRValue(self.new_id(), f"tmp_{self.new_id()}", result_type)
            ir_node = IRNode(self.new_id(), op, result, [left, right])
            self.nodes.append(ir_node)
            return result

        elif isinstance(node, ASTUnaryOp):
            operand = self.build_expression(node.operand)
            result = IRValue(self.new_id(), f"tmp_{self.new_id()}", operand.vtype)

            if node.op == '-':
                # -x => 0 - x
                zero = self.get_constant(0)
                ir_node = IRNode(self.new_id(), OpType.SUB, result, [zero, operand])
                self.nodes.append(ir_node)

            return result

        elif isinstance(node, ASTCall):
            return self.build_call(node)

        elif isinstance(node, ASTIndex):
            base = self.build_expression(node.base)
            index = self.build_expression(node.index)

            # Compute address
            addr = IRValue(self.new_id(), f"addr_{self.new_id()}", ValueType.SCALAR)
            addr_node = IRNode(self.new_id(), OpType.ADD, addr, [base, index])
            self.nodes.append(addr_node)

            # Load
            result = IRValue(self.new_id(), f"load_{self.new_id()}", ValueType.SCALAR)
            load_node = IRNode(self.new_id(), OpType.LOAD, result, [addr])
            self.nodes.append(load_node)
            return result

        else:
            raise ValueError(f"Unknown expression type: {type(node)}")

    def build_call(self, node: ASTCall) -> IRValue:
        """Build IR for function call."""
        if node.name == 'hash':
            # High-level hash operation
            arg = self.build_expression(node.args[0])
            result = IRValue(self.new_id(), f"hash_{self.new_id()}", arg.vtype)
            ir_node = IRNode(self.new_id(), OpType.HASH, result, [arg])
            self.nodes.append(ir_node)
            return result

        elif node.name == 'load':
            addr = self.build_expression(node.args[0])
            result = IRValue(self.new_id(), f"load_{self.new_id()}", ValueType.SCALAR)
            ir_node = IRNode(self.new_id(), OpType.LOAD, result, [addr])
            self.nodes.append(ir_node)
            return result

        elif node.name == 'vload':
            addr = self.build_expression(node.args[0])
            result = IRValue(self.new_id(), f"vload_{self.new_id()}", ValueType.VECTOR)
            ir_node = IRNode(self.new_id(), OpType.VLOAD, result, [addr])
            self.nodes.append(ir_node)
            return result

        elif node.name == 'broadcast':
            value = self.build_expression(node.args[0])
            result = IRValue(self.new_id(), f"broadcast_{self.new_id()}", ValueType.VECTOR)
            ir_node = IRNode(self.new_id(), OpType.VBROADCAST, result, [value])
            self.nodes.append(ir_node)
            return result

        elif node.name == 'store':
            addr = self.build_expression(node.args[0])
            value = self.build_expression(node.args[1])
            ir_node = IRNode(self.new_id(), OpType.STORE,
                           IRValue(self.new_id(), "_", ValueType.SCALAR), [addr, value])
            self.nodes.append(ir_node)
            return value

        elif node.name == 'vstore':
            addr = self.build_expression(node.args[0])
            value = self.build_expression(node.args[1])
            ir_node = IRNode(self.new_id(), OpType.VSTORE,
                           IRValue(self.new_id(), "_", ValueType.SCALAR), [addr, value])
            self.nodes.append(ir_node)
            return value

        else:
            raise ValueError(f"Unknown function: {node.name}")

    def build_for(self, node: ASTFor):
        """Build IR for loop (unrolled if vectorize hint)."""
        # For now, unroll the loop
        start = node.start.value if isinstance(node.start, ASTNumber) else 0
        end = node.end.value if isinstance(node.end, ASTNumber) else 1
        step = node.step.value if isinstance(node.step, ASTNumber) else 1

        for i in range(start, end, step):
            # Set iterator value
            iter_val = self.get_or_create_value(node.iterator)
            const_node = IRNode(self.new_id(), OpType.CONST, iter_val,
                              [self.get_constant(i)])
            self.nodes.append(const_node)

            # Build body
            for stmt in node.body:
                self.build_statement(stmt)

    def build_if(self, node: ASTIf):
        """Build IR for conditional."""
        cond = self.build_expression(node.condition)

        # For simple cases, we can use select
        # For complex cases, would need branch handling
        # Simplified: just build both branches and use select

        # Build true branch
        true_vals = {}
        for stmt in node.true_body:
            self.build_statement(stmt)
            if isinstance(stmt, ASTAssign) and isinstance(stmt.target, ASTIdentifier):
                true_vals[stmt.target.name] = self.values.get(stmt.target.name)

        # Build false branch
        false_vals = {}
        for stmt in node.false_body:
            self.build_statement(stmt)
            if isinstance(stmt, ASTAssign) and isinstance(stmt.target, ASTIdentifier):
                false_vals[stmt.target.name] = self.values.get(stmt.target.name)

    def build_var_decl(self, node: ASTVarDecl):
        """Build IR for variable declaration."""
        vtype = ValueType.VECTOR if node.vtype == 'vector' else ValueType.SCALAR
        self.values[node.name] = IRValue(self.new_id(), node.name, vtype)

    def build_decorator(self, node: ASTDecorator):
        """Handle decorators."""
        if node.name == 'vectorize':
            self.vectorize_hint = True
            if node.args:
                self.vectorize_size = node.args[0].value if isinstance(node.args[0], ASTNumber) else VLEN

        self.build_statement(node.target)

    def build_function(self, node: ASTFunction):
        """Build IR for function."""
        # Create parameters
        for param in node.params:
            self.get_or_create_value(param)

        # Build body
        for stmt in node.body:
            self.build_statement(stmt)


# ============== Scheduler ==============

class Scheduler:
    """Schedule IR nodes into VLIW instructions."""

    def __init__(self, nodes: List[IRNode]):
        self.nodes = nodes
        self.schedule: List[dict] = []
        self.node_to_cycle: Dict[int, int] = {}

    def get_engine(self, op: OpType) -> str:
        """Get engine for operation type."""
        if op in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.MOD,
                  OpType.XOR, OpType.AND, OpType.OR, OpType.SHL, OpType.SHR,
                  OpType.LT, OpType.EQ):
            return "alu"
        elif op == OpType.CONST:
            return "load"
        elif op in (OpType.LOAD, OpType.VLOAD):
            return "load"
        elif op in (OpType.STORE, OpType.VSTORE):
            return "store"
        elif op == OpType.VBROADCAST:
            return "valu"
        elif op in (OpType.SELECT, OpType.VSELECT):
            return "flow"
        elif op == OpType.MULTIPLY_ADD:
            return "valu"
        else:
            return "alu"  # Default

    def is_vector_op(self, node: IRNode) -> bool:
        """Check if node is a vector operation."""
        if node.op in (OpType.VLOAD, OpType.VSTORE, OpType.VBROADCAST,
                       OpType.VSELECT, OpType.MULTIPLY_ADD):
            return True
        if node.dest.vtype == ValueType.VECTOR:
            return True
        return False

    def build_dependencies(self) -> Dict[int, Set[int]]:
        """Build dependency graph (node id -> set of dependencies)."""
        deps = {node.id: set() for node in self.nodes}

        # Track last writer for each value
        last_write: Dict[int, int] = {}  # value id -> node id

        for node in self.nodes:
            # Add RAW dependencies
            for src in node.sources:
                if src.id in last_write:
                    deps[node.id].add(last_write[src.id])

            # Update last writer
            if node.dest:
                last_write[node.dest.id] = node.id

        return deps

    def schedule_list(self) -> List[dict]:
        """Schedule using list scheduling algorithm."""
        deps = self.build_dependencies()
        scheduled = set()
        ready = []

        # Find initially ready nodes (no dependencies)
        for node in self.nodes:
            if not deps[node.id]:
                ready.append(node)

        cycle = 0
        current_bundle: Dict[str, List[tuple]] = defaultdict(list)

        while ready or len(scheduled) < len(self.nodes):
            # Pack as many ready nodes as possible
            to_remove = []

            for node in ready:
                engine = self.get_engine(node.op)
                is_vec = self.is_vector_op(node)
                actual_engine = "valu" if is_vec and engine == "alu" else engine

                limit = SLOT_LIMITS.get(actual_engine, 1)

                if len(current_bundle[actual_engine]) < limit:
                    # Can schedule this node
                    slot = self.node_to_slot(node)
                    current_bundle[actual_engine].append(slot)
                    node.scheduled_cycle = cycle
                    node.engine = actual_engine
                    scheduled.add(node.id)
                    to_remove.append(node)

            for node in to_remove:
                ready.remove(node)

            # Emit current bundle if non-empty
            if current_bundle:
                self.schedule.append(dict(current_bundle))
                current_bundle = defaultdict(list)

            # Advance cycle
            cycle += 1

            # Find newly ready nodes
            for node in self.nodes:
                if node.id in scheduled:
                    continue
                if all(dep in scheduled for dep in deps[node.id]):
                    if node not in ready:
                        ready.append(node)

            # Safety: prevent infinite loop
            if cycle > len(self.nodes) * 10:
                break

        return self.schedule

    def node_to_slot(self, node: IRNode) -> tuple:
        """Convert IR node to VLIW slot tuple."""
        if node.op == OpType.CONST:
            return ("const", node.dest.id, node.sources[0].constant_value)

        elif node.op == OpType.LOAD:
            return ("load", node.dest.id, node.sources[0].id)

        elif node.op == OpType.VLOAD:
            return ("vload", node.dest.id, node.sources[0].id)

        elif node.op == OpType.STORE:
            return ("store", node.sources[0].id, node.sources[1].id)

        elif node.op == OpType.VSTORE:
            return ("vstore", node.sources[0].id, node.sources[1].id)

        elif node.op == OpType.VBROADCAST:
            return ("vbroadcast", node.dest.id, node.sources[0].id)

        elif node.op in IR_TO_VLIW:
            op_str = IR_TO_VLIW[node.op]
            return (op_str, node.dest.id, node.sources[0].id, node.sources[1].id)

        else:
            # Default binary op
            return ("+", node.dest.id, node.sources[0].id,
                   node.sources[1].id if len(node.sources) > 1 else 0)


# ============== Hash Expander ==============

class HashExpander:
    """Expand high-level hash operations into IR."""

    def __init__(self, builder: IRBuilder):
        self.builder = builder

    def expand(self, nodes: List[IRNode]) -> List[IRNode]:
        """Expand all hash operations."""
        result = []

        for node in nodes:
            if node.op == OpType.HASH:
                result.extend(self.expand_hash(node))
            else:
                result.append(node)

        return result

    def expand_hash(self, node: IRNode) -> List[IRNode]:
        """Expand hash(x) into the 6-stage hash function."""
        expanded = []
        val = node.sources[0]
        result_type = val.vtype

        # Create temporary values
        tmp1 = IRValue(self.builder.new_id(), f"htmp1_{node.id}", result_type)
        tmp2 = IRValue(self.builder.new_id(), f"htmp2_{node.id}", result_type)

        current_val = val

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Get constant values
            const1 = self.builder.get_constant(val1)
            const3 = self.builder.get_constant(val3)

            # Stage computation:
            # tmp1 = current_val op1 val1
            # tmp2 = current_val op3 val3
            # current_val = tmp1 op2 tmp2

            op1_type = OP_MAP.get(op1, OpType.ADD)
            op3_type = OP_MAP.get(op3, OpType.ADD)
            op2_type = OP_MAP.get(op2, OpType.ADD)

            # tmp1 = val op1 const1
            node1 = IRNode(self.builder.new_id(), op1_type, tmp1, [current_val, const1])
            expanded.append(node1)

            # tmp2 = val op3 const3
            node2 = IRNode(self.builder.new_id(), op3_type, tmp2, [current_val, const3])
            expanded.append(node2)

            # new_val = tmp1 op2 tmp2
            new_val = IRValue(self.builder.new_id(), f"hval_{node.id}_{stage_idx}", result_type)
            node3 = IRNode(self.builder.new_id(), op2_type, new_val, [tmp1, tmp2])
            expanded.append(node3)

            current_val = new_val

        # Copy final result to destination
        final_node = IRNode(self.builder.new_id(), OpType.ADD, node.dest,
                           [current_val, self.builder.get_constant(0)])
        expanded.append(final_node)

        return expanded


# ============== Vectorization Pass ==============

class Vectorizer:
    """Vectorize scalar operations."""

    def __init__(self, vlen: int = VLEN):
        self.vlen = vlen

    def vectorize(self, nodes: List[IRNode], batch_vars: Set[str]) -> List[IRNode]:
        """
        Vectorize operations on batch variables.

        Args:
            nodes: IR nodes to vectorize
            batch_vars: Variable names that should be vectorized
        """
        result = []

        for node in nodes:
            # Check if this operates on batch variables
            should_vectorize = False
            for src in node.sources:
                if src.name in batch_vars or src.vtype == ValueType.VECTOR:
                    should_vectorize = True
                    break

            if node.dest.name in batch_vars:
                should_vectorize = True

            if should_vectorize:
                # Convert to vector operation
                vectorized = self.vectorize_node(node)
                result.append(vectorized)
            else:
                result.append(node)

        return result

    def vectorize_node(self, node: IRNode) -> IRNode:
        """Convert scalar node to vector node."""
        new_node = deepcopy(node)
        new_node.dest.vtype = ValueType.VECTOR

        # Update sources that should be vectors
        for src in new_node.sources:
            if not src.is_constant:
                src.vtype = ValueType.VECTOR

        return new_node


# ============== Main Compiler ==============

class DSLCompiler:
    """Main DSL compiler."""

    def __init__(self):
        self.scratch_ptr = 0
        self.scratch_map: Dict[str, int] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def alloc_scratch(self, name: str, size: int = 1) -> int:
        """Allocate scratch memory."""
        addr = self.scratch_ptr
        self.scratch_map[name] = addr
        self.scratch_ptr += size
        return addr

    def compile(self, source: str, vectorize: bool = True) -> CompileResult:
        """
        Compile DSL source to VLIW instructions.

        Args:
            source: DSL source code
            vectorize: Whether to auto-vectorize

        Returns:
            CompileResult with instructions and metadata
        """
        try:
            # Lexing
            lexer = Lexer(source)
            tokens = lexer.tokenize()

            # Parsing
            parser = Parser(tokens)
            ast = parser.parse()

            # IR building
            builder = IRBuilder()
            ir_nodes = builder.build(ast)

            # Hash expansion
            expander = HashExpander(builder)
            ir_nodes = expander.expand(ir_nodes)

            # Vectorization (if enabled)
            if vectorize and builder.vectorize_hint:
                vectorizer = Vectorizer(builder.vectorize_size)
                # Find batch variables (those declared as vec)
                batch_vars = {name for name, val in builder.values.items()
                             if val.vtype == ValueType.VECTOR}
                ir_nodes = vectorizer.vectorize(ir_nodes, batch_vars)

            # Allocate scratch for all values
            for val in builder.values.values():
                size = VLEN if val.vtype == ValueType.VECTOR else 1
                if val.name not in self.scratch_map:
                    val.id = self.alloc_scratch(val.name, size)

            # Allocate scratch for constants
            for val in builder.constants.values():
                if val.name not in self.scratch_map:
                    val.id = self.alloc_scratch(val.name, 1)

            # Update node IDs with scratch addresses
            for node in ir_nodes:
                node.dest.id = self.scratch_map.get(node.dest.name, node.dest.id)
                for src in node.sources:
                    src.id = self.scratch_map.get(src.name, src.id)

            # Scheduling
            scheduler = Scheduler(ir_nodes)
            instructions = scheduler.schedule_list()

            return CompileResult(
                success=True,
                instructions=instructions,
                cycles=len(instructions),
                scratch_used=self.scratch_ptr,
                scratch_map=self.scratch_map,
                errors=self.errors,
                warnings=self.warnings,
                stats={
                    "ir_nodes": len(ir_nodes),
                    "values": len(builder.values),
                    "constants": len(builder.constants),
                }
            )

        except Exception as e:
            self.errors.append(str(e))
            return CompileResult(
                success=False,
                instructions=[],
                cycles=0,
                scratch_used=0,
                scratch_map={},
                errors=self.errors,
                warnings=self.warnings
            )

    def compile_to_kernel_builder(self, source: str) -> Tuple[List[dict], Dict[str, int]]:
        """
        Compile and return instructions compatible with KernelBuilder.

        Returns:
            Tuple of (instructions, scratch_map)
        """
        result = self.compile(source)
        if not result.success:
            raise ValueError(f"Compilation failed: {result.errors}")
        return result.instructions, result.scratch_map


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_result(self, result: CompileResult):
        print("=" * 70)
        print("DSL COMPILATION RESULT")
        print("=" * 70)
        print()
        print(f"Success:        {result.success}")
        print(f"Cycles:         {result.cycles}")
        print(f"Scratch used:   {result.scratch_used}")
        print()

        if result.errors:
            print("ERRORS:")
            for e in result.errors:
                print(f"  - {e}")
            print()

        if result.warnings:
            print("WARNINGS:")
            for w in result.warnings:
                print(f"  - {w}")
            print()

        if result.stats:
            print("STATISTICS:")
            for k, v in result.stats.items():
                print(f"  {k}: {v}")
            print()

        print("SCRATCH MAP:")
        for name, addr in sorted(result.scratch_map.items(), key=lambda x: x[1]):
            print(f"  {addr:4d}: {name}")
        print()

        print("INSTRUCTIONS:")
        for i, instr in enumerate(result.instructions[:20]):  # First 20
            print(f"  {i:4d}: {instr}")
        if len(result.instructions) > 20:
            print(f"  ... and {len(result.instructions) - 20} more")


class RichPrinter:
    """Rich-enabled colorful output."""

    def __init__(self):
        self.console = Console()

    def print_result(self, result: CompileResult):
        self.console.print(Panel("DSL COMPILATION RESULT", style="bold cyan", box=box.DOUBLE))

        # Summary table
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        success_style = "green" if result.success else "red"
        table.add_row("Success", f"[{success_style}]{result.success}[/{success_style}]")
        table.add_row("Cycles", f"{result.cycles:,}")
        table.add_row("Scratch used", f"{result.scratch_used:,}")

        self.console.print(table)

        # Errors
        if result.errors:
            self.console.print("\n[red bold]ERRORS:[/red bold]")
            for e in result.errors:
                self.console.print(f"  [red]-[/red] {e}")

        # Warnings
        if result.warnings:
            self.console.print("\n[yellow bold]WARNINGS:[/yellow bold]")
            for w in result.warnings:
                self.console.print(f"  [yellow]-[/yellow] {w}")

        # Stats
        if result.stats:
            self.console.print("\n[dim]Statistics:[/dim]")
            for k, v in result.stats.items():
                self.console.print(f"  [dim]{k}:[/dim] {v}")

        # Scratch map
        self.console.print("\n[bold]Scratch Allocation:[/bold]")
        scratch_table = Table(box=box.ROUNDED)
        scratch_table.add_column("Address", justify="right", style="cyan")
        scratch_table.add_column("Name", style="green")

        for name, addr in sorted(result.scratch_map.items(), key=lambda x: x[1])[:15]:
            scratch_table.add_row(str(addr), name)

        if len(result.scratch_map) > 15:
            scratch_table.add_row("...", f"({len(result.scratch_map) - 15} more)")

        self.console.print(scratch_table)

        # Instructions
        self.console.print("\n[bold]Instructions:[/bold]")
        for i, instr in enumerate(result.instructions[:10]):
            self.console.print(f"  [dim]{i:4d}:[/dim] {instr}")
        if len(result.instructions) > 10:
            self.console.print(f"  [dim]... and {len(result.instructions) - 10} more[/dim]")


def get_printer(use_color: bool = True):
    """Get appropriate printer."""
    if use_color and RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Example DSL ==============

EXAMPLE_DSL = '''
# Example DSL for hash+tree algorithm
# This shows the basic structure

# Declare variables
var idx
var val
var node_val

# Vector version (batch of 8)
vec v_idx
vec v_val
vec v_node_val

@vectorize(8)
def process_batch(base_idx, base_val, forest_p):
    # Load batch data
    v_idx = vload(base_idx)
    v_val = vload(base_val)

    # Compute tree addresses
    v_addr = v_idx + forest_p

    # Hash computation (expands to 6 stages)
    v_mixed = v_val ^ v_node_val
    v_hashed = hash(v_mixed)

    # Index update
    v_new_idx = v_idx * 2 + 1

    # Store results
    vstore(base_val, v_hashed)
    vstore(base_idx, v_new_idx)
'''


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="DSL Compiler for VLIW SIMD Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/dsl_compiler/dsl_compiler.py example.dsl
    python tools/dsl_compiler/dsl_compiler.py example.dsl --json
    python tools/dsl_compiler/dsl_compiler.py --demo
        """
    )
    parser.add_argument("file", nargs="?", help="DSL source file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--demo", action="store_true", help="Run with example DSL")
    parser.add_argument("--no-vectorize", action="store_true", help="Disable auto-vectorization")
    args = parser.parse_args()

    # Get source
    if args.demo:
        source = EXAMPLE_DSL
    elif args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        # Read from stdin or show help
        print("Usage: python dsl_compiler.py <file.dsl> [--json] [--demo]")
        print("\nExample DSL syntax:")
        print(EXAMPLE_DSL)
        return

    # Compile
    compiler = DSLCompiler()
    result = compiler.compile(source, vectorize=not args.no_vectorize)

    # Output
    if args.json:
        print(result.to_json())
    else:
        printer = get_printer(not args.no_color)
        printer.print_result(result)


if __name__ == "__main__":
    main()
