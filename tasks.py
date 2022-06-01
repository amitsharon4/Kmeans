from invoke import task, call


@task
def clear(c):
    c.run("clear")


@task(clear)
def delete(c, extra=''):
    patterns = ['*mykmeans*.so']
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))

@task(delete)
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(build)
def run(c, k=-1, n=-1, Random=True):
    if Random:
        c.run("python3.8.5 main.py {} {} --random" .format(k, n))
    else:
        c.run("python3.8.5 main.py {} {} --no-random" .format(k, n))
