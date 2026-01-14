class LossTypeRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.loss_type.name
            return super().__getitem__(name)(key)
        return super().__getitem__(key)

    def __call__(self, config):
        return self[config]


_LOSSTYPE = LossTypeRegistry()


def register_loss_type(cls=None, *, name=None):
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _LOSSTYPE:
            raise ValueError(f'Already registered loss type with name: {local_name}')
        _LOSSTYPE[local_name] = cls
        return cls

    return _register(cls) if cls is not None else _register