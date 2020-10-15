#pragma once

#include <thread>

extern std::thread workerThread;

void worker_init();
void worker_finalize();