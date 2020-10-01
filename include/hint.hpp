#pragma once

struct Hint {
    bool valid;
    const char *name;
    int line;

    enum Optimization {NONE, REPEATED};
    Optimization opt;
};

extern "C" Hint hint;


void hint_clear();
void hint_repeated(const char *name, int lineno);


#define HINT_REPEATED() hint_repeated(__FILE__, __LINE__)
#define HINT_CLEAR() hint_clear()