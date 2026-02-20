"""Optional TLS support for gRPC servers and clients."""

import grpc


def make_server_credentials(cert_path: str, key_path: str) -> grpc.ServerCredentials:
    """Load PEM cert + key and return gRPC server credentials."""
    with open(key_path, "rb") as f:
        private_key = f.read()
    with open(cert_path, "rb") as f:
        certificate_chain = f.read()
    return grpc.ssl_server_credentials([(private_key, certificate_chain)])


def configure_server_port(
    server: grpc.Server,
    host: str,
    port: int,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> int:
    """Bind server to a port — secure if certs provided, insecure otherwise.

    Returns the port number actually bound.
    """
    address = f"{host}:{port}"
    if tls_cert and tls_key:
        creds = make_server_credentials(tls_cert, tls_key)
        bound = server.add_secure_port(address, creds)
        print(f"[TLS] Secure port bound: {address}")
    else:
        bound = server.add_insecure_port(address)
    return bound


def make_channel_credentials(ca_cert_path: str) -> grpc.ChannelCredentials:
    """Load a CA certificate and return gRPC channel credentials."""
    with open(ca_cert_path, "rb") as f:
        root_cert = f.read()
    return grpc.ssl_channel_credentials(root_certificates=root_cert)
