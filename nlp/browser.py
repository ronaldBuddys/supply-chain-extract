

from selenium import webdriver
from selenium.webdriver import FirefoxOptions


def start_browser(profile_loc=None, out_dir=None, width=None, height=None):

    # load profile - see (in linux): /home/<user>/.mozilla/firefox/<profile_name>
    profile = webdriver.FirefoxProfile(profile_loc)
    profile.set_preference("browser.download.folderList", 2)
    # profile.set_preference("browser.download.manager.showWhenStarting", False)
    if out_dir is not None:
        profile.set_preference("browser.download.dir", out_dir)

    # TODO: change this to properly handle not asking to save xlsx
    #  - why doesn't following work :'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ?
    # https://stackoverflow.com/questions/4212861/what-is-a-correct-mime-type-for-docx-pptx-etc/4212908#4212908
    # source: https://stackoverflow.com/questions/59818982/how-to-overcome-firefox-prompt-to-save-file
    mime_types = [
        'text/plain',
        'attachment/vnd.ms-excel',
        'text/csv',
        'application/csv',
        'text/comma-separated-values',
        'application/download',
        'application/octet-stream',
        'binary/octet-stream',
        'application/binary',
        'application/x-unknown',
        'application/excel',
        'attachment/csv',
        'attachment/excel'
        'application/vnd.ms-excel',
        'application/msexcel',
        'application/x-msexcel',
        'application/x-ms-excel',
        'application/x-excel',
        'application/x-dos_ms_excel',
        'application/xls',
        'application/x-xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ]

    # NOTE: this is not working as expected - needed to change in
    # FireFox -> (hamburger, 3 horizontal lines) -> (scroll down to) Applications ->
    # select action for Microsoft Excel Worksheet: Save File
    # - this needs to be done in the profile before running with selenium (profile gets copied?)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk",
                            ",".join(mime_types))
    profile.set_preference('media.mp4.enabled', False)

    # user_agent = 'Mozilla/5.0'
    # profile.set_preference("general.useragent.override", user_agent)
    # set options for width and height
    opts = FirefoxOptions()
    if isinstance(width, int):
        opts.add_argument(f"--width={width}")
    if isinstance(height, int):
        opts.add_argument(f"--height={height}")

    # this one seems to work - requires profile
    PROXY_HOST = "12.12.12.123"
    PROXY_PORT = "1234"
    profile.set_preference("network.proxy.type", 1)
    profile.set_preference("network.proxy.http", PROXY_HOST)
    profile.set_preference("network.proxy.http_port", int(PROXY_PORT))
    profile.set_preference("dom.webdriver.enabled", False)
    profile.set_preference('useAutomationExtension', False)
    profile.update_preferences()
    desired = webdriver.DesiredCapabilities.FIREFOX

    browser = webdriver.Firefox(firefox_profile=profile, desired_capabilities=desired)

    return browser


if __name__ == "__main__":

    # start browser - without profile
    browser = start_browser()

    # begin to surf the web
    browser.get("https://www.google.com")
