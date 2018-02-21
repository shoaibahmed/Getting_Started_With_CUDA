
inline int _ConvertSMVer2Cores(int major, int minor)
{
    int index=0;
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x60, 128}, // Pascal Generation (SM 6.1) 1080ti class
        { 0x61, 128}, // Pascal Generation (SM 6.1) 1080ti class
        {   -1, -1 }
    };

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    return 0;
}
inline int get_chunk_env() {
 int def=-1;
 char *env;
 env = getenv("GPU_CHUNK");
 if(env) {
  printf("Using %d chunksize instead of %d\n",atoi(env),def);
  def=atoi(env);
 }
 return(def);
}

inline int get_threads_env() {
 int def=-1;
 char *env;
 env = getenv("GPU_THREADS");
 if(env) {
  printf("Using %d threads instead of %d\n",atoi(env),def);
  def=atoi(env);
 }
 return(def);
}
