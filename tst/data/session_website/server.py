"""A website on WebsiteEnvironment with no pages of its own, named shop.test, for integration-testing browser_sign_in:
website/v1's session signs a browser in, and /api/session says who a request's cookie signs in."""
from agentenv_protocol.website import WebsiteEnvironment, website


@website(domains=["shop.test"])
class Shop(WebsiteEnvironment):
    pass


if __name__ == "__main__":
    Shop().serve()
