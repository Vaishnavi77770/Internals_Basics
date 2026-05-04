import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--attack_surface_count", type=int, required=True)
parser.add_argument("--patch_age_days", type=int, required=True)
parser.add_argument("--is_external_facing", type=int, required=True)
parser.add_argument("--tech_stack_complexity", type=int, required=True)

args = parser.parse_args()

data = pd.DataFrame([[
    args.attack_surface_count,
    args.patch_age_days,
    args.is_external_facing,
    args.tech_stack_complexity
]], columns=[
    "attack_surface_count",
    "patch_age_days",
    "is_external_facing",
    "tech_stack_complexity"
])

prediction = data.mean(axis=1)[0]

print(prediction)