#include "packer_cache.hpp"

/*extern*/ std::map<MPI_Datatype, std::unique_ptr<Packer>> packerCache;