

def raw_data_stats(data):
    help_count = 0
    rule_five_count = 0
    normal_count = 0

    for dp in data:
        if dp["help"] and not dp["rule5"]:
            help_count += 1
        elif dp["rule5"]:
            rule_five_count += 1
        else:
            normal_count += 1

    total = help_count + rule_five_count + normal_count

    print(f"There are \t{help_count}\t help samples \t\t({(help_count / total) * 100:.2f} %)")
    print(f"\t\t{rule_five_count}\t rule five samples \t({(rule_five_count / total) * 100:.2f} %)")
    print(f"\t\t{normal_count}\t normal samples \t({(normal_count / total) * 100:.2f} %)")

    print(f"\nTotal: {total}")
    print("\n")


def separate_data(data):
    texts = []
    labels = []

    for dp in data:
        texts.append(dp["text"])

        help_dp = dp["help"]
        rule5_dp = dp["rule5"]

        if rule5_dp or help_dp:
            labels.append([0, int(help_dp), int(rule5_dp)])
        else:
            labels.append([1, 0, 0])

    return texts, labels
