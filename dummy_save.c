#include "src/model/model.h"
int main() { Model m; model_init(&m); model_save(&m, "model.bin"); return 0; }
