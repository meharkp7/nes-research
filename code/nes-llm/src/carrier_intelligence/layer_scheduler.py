class LayerScheduler:
    """
    Phase 3

    Capacity-Constrained Layer Scheduler.

    Builds a payload allocation plan from
    layer quality statistics.

    The scheduler DOES NOT receive the
    payload itself.

    It only decides how much of the future
    payload should be assigned to each
    transformer layer.
    """

    def __init__(
        self,
        minimum_fraction=0.01,
    ):

        self.minimum_fraction = minimum_fraction

    def build_plan(
        self,
        layer_profiles,
    ):

        total_score = sum(

            profile["mean_quality"] *
            profile["carrier_count"]

            for profile in layer_profiles

        )

        if total_score == 0:

            raise ValueError(
                "Layer quality is zero."
            )

        plan = []

        for profile in layer_profiles:

            score = (

                profile["mean_quality"]

                *

                profile["carrier_count"]

            )

            weight = score / total_score

            weight = max(

                weight,

                self.minimum_fraction,

            )

            plan.append(

                {

                    "layer":

                    profile["layer"],

                    "weight":

                    weight,

                    "quality":

                    profile["mean_quality"],

                    "capacity":

                    profile["carrier_count"],

                }

            )

        normalization = sum(

            layer["weight"]

            for layer in plan

        )

        for layer in plan:

            layer["weight"] /= normalization

        return plan

    def allocate_payload(
        self,
        payload_bits,
        allocation_plan,
    ):

        payload_size = len(
            payload_bits
        )

        chunks = []

        cursor = 0

        for idx, layer in enumerate(
            allocation_plan
        ):

            if idx == len(
                allocation_plan
            ) - 1:

                end = payload_size

            else:

                end = (

                    cursor

                    +

                    int(

                        payload_size

                        *

                        layer["weight"]

                    )

                )

            chunks.append(

                {

                    "layer":

                    layer["layer"],

                    "bits":

                    payload_bits[
                        cursor:end
                    ],

                    "capacity":

                    layer["capacity"],

                    "quality":

                    layer["quality"],

                }

            )

            cursor = end

        return chunks


def main():

    scheduler = LayerScheduler()

    profiles = [

        {

            "layer": 0,

            "mean_quality": 0.81,

            "carrier_count": 5000,

        },

        {

            "layer": 1,

            "mean_quality": 0.64,

            "carrier_count": 4200,

        },

        {

            "layer": 2,

            "mean_quality": 0.22,

            "carrier_count": 3800,

        },

    ]

    plan = scheduler.build_plan(
        profiles
    )

    print("\nAllocation Plan\n")

    for p in plan:

        print(p)

    payload = [1] * 10000

    chunks = scheduler.allocate_payload(
        payload,
        plan,
    )

    print("\nPayload Distribution\n")

    for c in chunks:

        print(

            c["layer"],

            len(c["bits"])

        )


if __name__ == "__main__":
    main()