"""
Only to be run in CI/CD.
Authors: Luca Mingarelli.
"""
import os

API_TOKEN = os.environ['CICD_API_TOKEN']
CI_PROJECT_ID = os.environ['CI_PROJECT_ID']

VERSION = (os.popen("""grep "__version__" mtalg/__about__.py | cut -d "'" -f 2""")
           .read()
           .strip('\n'))

url = f'https://gitlab.sofa.dev/api/v4/projects/{CI_PROJECT_ID}/badges'

BADGES = os.popen(f'curl  --header "PRIVATE-TOKEN:{API_TOKEN}" {url}').read()
exec(f'BADGES = {BADGES}')

VERSION_BADGE_ID = [b['id'] for b in BADGES if b['name'] == 'version'][0]

with open('VERSION_BADGE_ID.txt', 'w') as f:
    f.write(f"version_badge_id = '{VERSION_BADGE_ID}'")

url_out = f'{url}/{VERSION_BADGE_ID}'
img_url = f'https://img.shields.io/badge/version-{VERSION}-brightgreen'

os.system(f'curl --request PUT --header "PRIVATE-TOKEN:{API_TOKEN}" --data "image_url={img_url}" {url_out}')
