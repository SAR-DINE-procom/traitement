wget https://www.mathworks.com/mpm/glnxa64/mpm && \
chmod +x mpm && \
./mpm install --release=R2025a --destination=/opt/matlab --products=MATLAB && \
ln -fs /opt/matlab/bin/matlab /usr/local/bin/matlab && \
MATLAB_DEPS_URL="https://raw.githubusercontent.com/mathworks-ref-arch/container-images/main/matlab-deps/r2025a/ubuntu22.04/base-dependencies.txt" && \
MATLAB_DEPENDENCIES=base-dependencies.txt && \
wget ${MATLAB_DEPS_URL} -O ${MATLAB_DEPENDENCIES} && \
xargs -a ${MATLAB_DEPENDENCIES} -r apt-get install --no-install-recommends -y && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/matlab/bin/glnxa64 && \
python3 -m pip install jupyter-matlab-proxy matlabengine==25.1.2 && \
env MWI_APP_PORT=3000 MWI_ENABLE_AUTH_TOKEN=False matlab-proxy-app && \
./mpm install --release=R2025a --destination=/opt/matlab --products=Phased_Array_System_Toolbox &