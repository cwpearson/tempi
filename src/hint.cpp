#include "hint.hpp"

/*extern "C"*/ Hint hint;

void hint_clear()
{
    hint = {};
}

void hint_repeated(const char *name, int lineno)
{
    hint.valid = true;
    hint.name = name;
    hint.line = lineno;
    hint.opt = Hint::Optimization::REPEATED;
}