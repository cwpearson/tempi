if the ssl eror is in urllib, you need to modify the poetry install script

```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

otherwise if its in Requests somewhere

```
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
```

## blake

```
salloc -N 1 --time=02:00:00
. load-env.sh
mpirun -n 2 poetry run python3 test.py
```