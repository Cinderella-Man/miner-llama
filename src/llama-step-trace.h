#pragma once

// LLAMA_STEP_TRACE=<file> (C405): host timestamps along the server's steps, for the time of a speculative step that no
// card's clock holds (the API is llama_step_trace_* in llama-ext.h). The contexts' side: whether the variable is set
// (read once), and a mark of a context's stage, which the step's line prints as "<role>.<name>".

bool llama_step_trace_active();
void llama_step_trace_put(char role, const char * name);
