"""
Test if the METEOR java tool works. Meteor is java-based and crashes alot depending on install java version.
"""
from pycocoevalcap.meteor.meteor import Meteor


def check_meteor_works():
    try:
        met = Meteor()
    except (AttributeError, FileNotFoundError) as e:
        print(f"Meteor couldn't start due to {e}")
        met = None

    gts = {
        "datapoint1": ["hello my name is", "meteor test program"],
        "datapoint2": ["another test sentence", "this the end of the test."]
    }
    refs = {
        "datapoint1": ["is my name really meteor"],
        "datapoint2": ["probably another test sentence"]
    }
    try:
        output = met.compute_score(gts, refs)
    except (ValueError, FileNotFoundError, AttributeError) as e:
        print(f"{e.__class__.__name__}: {e}")
        met.lock.release()
        return False
    print(output)
    return True


def main():
    works = check_meteor_works()
    print(f"Meteor works: {works}.")


if __name__ == "__main__":
    main()
