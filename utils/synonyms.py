import copy


class SimilarCategories(object):
    """This class helps find similar categories.
    """

    def __init__(
        self,
        object_alias_file="data/VisualGenome/object_alias.txt",
        predicate_alias_file="data/VisualGenome/relationship_alias.txt",
    ):
        """Constructor for SimilarCategories.

        Args:
            object_alias_file: A file containing objects on every line with
                each line containing a comma (,) separated list of categories.
            predicate_alias_file: A file containing predicates on every line
                with each line containing a comma (,) separated list of
                categories.
        """
        self.object_alias = self._gather_similar_categories(object_alias_file)
        self.predicate_alias = self._gather_similar_categories(predicate_alias_file)

    def _gather_similar_categories(self, alias_file):
        """Organizes the categories in the file accordingly.

        Args:
            alias_file: file containing categories on every line with
                each line containing a comma (,) separated list of categories.

        Returns:
            A dictionary from the words to its categories.
        """
        alias_map = {}
        f = open(alias_file, "r")
        for line in f:
            cats = line.strip().split(",")
            for cat in cats:
                for word in cat.split(" "):
                    if word not in alias_map:
                        alias_map[word] = []
                    alias_map[word].extend(cats)
        return alias_map

    def _get_similar_categories(self, seed_list, alias_map):
        """Gets the aliases of the categories in the seed list.

        Args:
            seed_list: A list of initial categories.
            alias_map: A dictionary from a category to its categories.

        Returns:
            A complete list of categories.
        """
        prev_set = set(seed_list)
        curr_set = set([])
        while prev_set != curr_set:
            for cat in prev_set:
                for word in cat.split(" "):
                    if word in alias_map:
                        curr_set.update(alias_map[word])
            prev_set = copy.deepcopy(curr_set)
        return list(curr_set)

    def get_similar_objects(self, seed_list):
        """Gets the aliases of the object in the seed list.

        Args:
            seed_list: A list of initial objects.
            alias_map: A dictionary from a object to its categories.

        Returns:
            A complete list of similar categories.
        """
        return self._get_similar_categories(seed_list, self.object_alias)

    def get_similar_predicates(self, seed_list):
        """Gets the aliases of the predicates in the seed list.

        Args:
            seed_list: A list of initial predicates.
            alias_map: A dictionary from a predicate to its categories.

        Returns:
            A complete list of similar categories.
        """
        return self._get_similar_categories(seed_list, self.predicate_alias)
