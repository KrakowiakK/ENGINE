#!/usr/bin/env python3
"""Expose the production engine (127.0.0.1:8099) on the local network WITHOUT restarting it: a plain TCP relay.

Bytes are copied both ways unchanged (HTTP, SSE streaming, keep-alive all pass through). Only peers from private
address ranges (RFC 1918, loopback, link-local) are accepted; every connection is logged with its peer and byte counts.
No authentication: anyone on the LAN can use the engine while this runs.

  nohup python3 tools/lan_proxy.py --listen <LAN_IP>:8099 > lan-proxy.log 2>&1 &
  kill <pid>          (stops the relay; production is untouched either way)
"""
import argparse
import asyncio
import ipaddress
import time


def allowed(peer: str) -> bool:
    ip = ipaddress.ip_address(peer)
    return ip.is_private or ip.is_loopback or ip.is_link_local


async def pipe(reader, writer, count):
    try:
        while data := await reader.read(65536):
            count[0] += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.write_eof() if writer.can_write_eof() else writer.close()
        except (OSError, RuntimeError):
            pass


async def handle(creader, cwriter, target):
    peer = cwriter.get_extra_info('peername')[0]
    if not allowed(peer):
        print(f'{time.strftime("%H:%M:%S")} REFUSED {peer}', flush=True)
        cwriter.close()
        return
    try:
        treader, twriter = await asyncio.open_connection(*target)
    except OSError as e:
        print(f'{time.strftime("%H:%M:%S")} {peer} engine unreachable: {e}', flush=True)
        cwriter.close()
        return
    up, down, t0 = [0], [0], time.time()
    await asyncio.gather(pipe(creader, twriter, up), pipe(treader, cwriter, down))
    for w in (cwriter, twriter):
        w.close()
    print(f'{time.strftime("%H:%M:%S")} {peer} {time.time() - t0:.1f}s up {up[0]} B down {down[0]} B', flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--listen', required=True, help='LAN_IP:PORT')
    ap.add_argument('--target', default='127.0.0.1:8099')
    a = ap.parse_args()
    lh, lp = a.listen.rsplit(':', 1)
    th, tp = a.target.rsplit(':', 1)
    server = await asyncio.start_server(lambda r, w: handle(r, w, (th, int(tp))), lh, int(lp))
    print(f'{time.strftime("%H:%M:%S")} relay {a.listen} -> {a.target}', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
