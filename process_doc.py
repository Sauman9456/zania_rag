import pymupdf4llm


def parse_doc(document_path):
    # Initialize variables
    text = pymupdf4llm.to_markdown(document_path)

    result = []
    parent_title = child_title = grand_child_title = great_grand_child_title = None
    current_content = ""

    # Split text by lines and process each line
    lines = text.splitlines()
    for line in lines:
        # Identify the level based on the number of hashtags at the beginning
        line = line.strip()
        if line.startswith("# "):
            # Save current entry if there's accumulated content
            if parent_title:
                result.append(
                    {
                        "parent_title": parent_title,
                        "child_title": child_title,
                        "grand_child_title": grand_child_title,
                        "great_grand_child_title": great_grand_child_title,
                    }
                )
            # Update to new parent title and reset all sub-levels
            parent_title = line
            child_title = grand_child_title = great_grand_child_title = None
        elif line.startswith("## "):
            # Save current entry if there's accumulated content at child level
            if child_title:
                result.append(
                    {
                        "parent_title": parent_title,
                        "child_title": child_title,
                        "grand_child_title": grand_child_title,
                        "great_grand_child_title": great_grand_child_title,
                    }
                )
            # Update to new child title and reset lower levels
            child_title = line
            grand_child_title = great_grand_child_title = None
        elif line.startswith("### "):
            # Save current entry if there's accumulated content at grandchild level
            if grand_child_title:
                result.append(
                    {
                        "parent_title": parent_title,
                        "child_title": child_title,
                        "grand_child_title": grand_child_title,
                        "great_grand_child_title": great_grand_child_title,
                    }
                )
            # Update to new grandchild title and reset lower level
            grand_child_title = line
            great_grand_child_title = None
        elif line.startswith("#### "):
            # Save current entry if there's accumulated content at great-grandchild level
            if great_grand_child_title:
                result.append(
                    {
                        "parent_title": parent_title,
                        "child_title": child_title,
                        "grand_child_title": grand_child_title,
                        "great_grand_child_title": great_grand_child_title,
                    }
                )
            # Update to new great-grandchild title
            great_grand_child_title = line
        else:
            # Add content to the current level if there is no heading
            if great_grand_child_title:
                great_grand_child_title += f"\n{line}"
            elif grand_child_title:
                grand_child_title += f"\n{line}"
            elif child_title:
                child_title += f"\n{line}"
            elif parent_title:
                parent_title += f"\n{line}"

    # Append the last item if there is any remaining content
    if parent_title:
        result.append(
            {
                "parent_title": parent_title,
                "child_title": child_title,
                "grand_child_title": grand_child_title,
                "great_grand_child_title": great_grand_child_title,
            }
        )

    return result
