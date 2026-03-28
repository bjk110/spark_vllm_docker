#!/usr/bin/env python3
"""
vLLM patch: Qwen3.5 MoE text-only compatibility shim.

Creates a text-only subclass of Qwen3_5MoeForConditionalGeneration that:
- Reuses the wrapper's hybrid cache-spec calculation (fixes page-size bug)
- Skips vision encoder initialization entirely
- Sets supports_multimodal = False (prevents multimodal warmup)
- Registers as Qwen3_5MoeForCausalLM in the model registry
"""

import re
import textwrap

REGISTRY_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
QWEN35_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"


def patch_registry():
    """Map Qwen3_5MoeForCausalLM -> Qwen3_5MoeTextOnlyShim."""
    with open(REGISTRY_PATH) as f:
        content = f.read()

    entry = '"Qwen3_5MoeForCausalLM"'
    if entry in content:
        # Update existing entry
        content = re.sub(
            r'"Qwen3_5MoeForCausalLM": \(\s*"qwen3_5",\s*"[^"]+",\s*\)',
            '"Qwen3_5MoeForCausalLM": (\n        "qwen3_5",\n        "Qwen3_5MoeTextOnlyShim",\n    )',
            content,
        )
        print("[patch] registry: Updated Qwen3_5MoeForCausalLM -> TextOnlyShim")
    else:
        # Add new entry
        target = '"Qwen3_5MoeForConditionalGeneration": (\n        "qwen3_5",\n        "Qwen3_5MoeForConditionalGeneration",\n    ),'
        insert = target + '\n    "Qwen3_5MoeForCausalLM": (\n        "qwen3_5",\n        "Qwen3_5MoeTextOnlyShim",\n    ),'
        if target in content:
            content = content.replace(target, insert)
        else:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '"Qwen3_5MoeForConditionalGeneration"' in line:
                    for j in range(i, min(i+5, len(lines))):
                        if lines[j].strip() == '),':
                            lines.insert(j+1, '    "Qwen3_5MoeForCausalLM": (')
                            lines.insert(j+2, '        "qwen3_5",')
                            lines.insert(j+3, '        "Qwen3_5MoeTextOnlyShim",')
                            lines.insert(j+4, '    ),')
                            content = '\n'.join(lines)
                            break
                    break
        print("[patch] registry: Added Qwen3_5MoeForCausalLM -> TextOnlyShim")

    with open(REGISTRY_PATH, 'w') as f:
        f.write(content)


def patch_add_text_only_shim():
    """Add Qwen3_5MoeTextOnlyShim class to qwen3_5.py."""
    with open(QWEN35_PATH) as f:
        content = f.read()

    if "Qwen3_5MoeTextOnlyShim" in content:
        print("[patch] qwen3_5: TextOnlyShim already exists")
        return

    # The shim class: inherits ConditionalGeneration for cache-spec,
    # but overrides __init__ to skip vision, and sets supports_multimodal = False
    shim_code = '''

########################################################
# Text-only compatibility shim
# Reuses ConditionalGeneration's cache-spec but skips vision
########################################################


class Qwen3_5MoeTextOnlyShim(Qwen3_5MoeForConditionalGeneration):
    """Text-only shim for Qwen3.5 MoE CausalLM checkpoints.

    Inherits Qwen3_5MoeForConditionalGeneration for hybrid cache-spec
    calculation (fixing the page-size bug in CausalLM path), but:
    - Does NOT initialize vision encoder
    - Does NOT register as multimodal
    - Rejects multimodal input at forward time
    """

    # Override: NOT a multimodal model
    supports_multimodal = False

    def __init__(self, *, vllm_config, prefix: str = "model"):
        import logging
        log = logging.getLogger("qwen3_5_text_only_shim")

        # Skip the parent's multimodal __init__ entirely
        # Go directly to nn.Module.__init__
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        self.config = config

        # vision_config is now a dummy with safe values (hidden_size=128)
        # created by the patched Qwen3_5MoeConfig.__init__
        vc = getattr(config, "vision_config", None)
        if vc is not None:
            log.info(f"vision_config present: hidden_size={getattr(vc, 'hidden_size', '?')}")

        # Inject dummy MultiModalConfig so _mark_language_model works
        # All defaults are safe for text-only (mm_encoder_only=False, etc.)
        from vllm.config.multimodal import MultiModalConfig
        if vllm_config.model_config.multimodal_config is None:
            vllm_config.model_config.multimodal_config = MultiModalConfig()
            log.info("Injected dummy MultiModalConfig for text-only shim")

        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.visual = None
        self.use_data_parallel = False
        self.is_multimodal_pruning_enabled = False
        self._text_only_mode = True

        log.info("Qwen3_5MoeTextOnlyShim: text-only mode, vision encoder skipped")

        # Use _mark_language_model to preserve wrapper's cache-spec path
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()

    def forward(self, *args, **kwargs):
        """Forward: delegate to language_model, reject multimodal input."""
        if kwargs.get("pixel_values") is not None or kwargs.get("image_grid_thw") is not None:
            raise ValueError(
                "Qwen3_5MoeTextOnlyShim does not support multimodal input. "
                "This model was loaded as text-only."
            )
        return self.language_model(*args, **kwargs)

    def load_weights(self, weights):
        """Load weights with key remapping for text-only checkpoints."""
        import logging
        log = logging.getLogger("qwen3_5_text_only_shim")

        def _remap(weights_iter):
            remapped = False
            for name, tensor in weights_iter:
                new_name = name
                # ModelOpt export uses model.language_model.* prefix
                # which matches our module tree (self.language_model.model.*)
                # No remapping needed for ConditionalGeneration path
                # since self.language_model prefix is already "language_model"
                yield new_name, tensor

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp.", "visual."])
        return loader.load_weights(_remap(weights), mapper=self.hf_to_vllm_mapper)

'''

    # Insert before the final class or at the end of file
    # Find the last class definition to insert after
    insert_pos = content.rfind('\nclass Qwen3_5MoeForConditionalGeneration')
    if insert_pos == -1:
        # Append at end
        content += shim_code
    else:
        # Find the end of Qwen3_5MoeForConditionalGeneration class (next class or EOF)
        # Insert after the entire ConditionalGeneration class
        # Find the set_moe_parameters() call which is the last line of __init__
        end_of_class = content.find('\nclass ', insert_pos + 10)
        if end_of_class == -1:
            content += shim_code
        else:
            # Actually, insert at the very end of the file
            content += shim_code

    with open(QWEN35_PATH, 'w') as f:
        f.write(content)
    print("[patch] qwen3_5: Added Qwen3_5MoeTextOnlyShim class")


def patch_processing_info():
    """Patch ProcessingInfo to handle text-only config gracefully."""
    with open(QWEN35_PATH) as f:
        content = f.read()

    if "text_only_shim_processing" in content:
        print("[patch] qwen3_5: ProcessingInfo already patched")
        return

    old = '''class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)'''

    new = '''class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    # text_only_shim_processing
    def get_hf_config(self):
        try:
            return self.ctx.get_hf_config(Qwen3_5MoeConfig)
        except TypeError:
            return self.ctx.model_config.hf_config

    def get_data_parser(self):
        config = self.get_hf_config()
        if not hasattr(config, "vision_config") or config.vision_config is None:
            from vllm.multimodal.parse import MultiModalDataParser
            return MultiModalDataParser()
        return super().get_data_parser()

    def get_max_image_tokens(self):
        config = self.get_hf_config()
        if not hasattr(config, "vision_config") or config.vision_config is None:
            return 0
        return super().get_max_image_tokens()

    def get_max_video_tokens(self, seq_len, mm_counts=None):
        config = self.get_hf_config()
        if not hasattr(config, "vision_config") or config.vision_config is None:
            return 0
        return super().get_max_video_tokens(seq_len, mm_counts)

    def get_image_size_with_most_features(self):
        config = self.get_hf_config()
        if not hasattr(config, "vision_config") or config.vision_config is None:
            return (0, 0)
        return super().get_image_size_with_most_features()'''

    if old in content:
        content = content.replace(old, new)
        with open(QWEN35_PATH, 'w') as f:
            f.write(content)
        print("[patch] qwen3_5: ProcessingInfo patched")
    else:
        print("[patch] qwen3_5: ProcessingInfo already modified or not found")


def patch_config_vision_default():
    """Prevent Qwen3_5MoeConfig from auto-creating vision_config when None.

    The original code: if vision_config is None -> create default VisionConfig.
    We change it: if vision_config is None -> keep as None.
    This prevents vision hidden_size=1152 from leaking into FP8 TP2 validation.
    """
    CONFIG_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"

    with open(CONFIG_PATH) as f:
        content = f.read()

    if "text_only_shim_config" in content:
        print("[patch] config: vision_config default already patched")
        return

    old = '        elif vision_config is None:\n            self.vision_config = self.sub_configs["vision_config"]()'
    # Instead of None, use a dummy with minimal safe values
    # This prevents NoneType errors in multimodal processing code
    # while keeping hidden_size small enough for TP2 block validation
    new = '''        elif vision_config is None:
            # text_only_shim_config: create minimal dummy vision config
            # with safe values that pass TP2 block-wise FP8 validation
            # hidden_size=128 is divisible by block_size=128 and any TP
            self.vision_config = self.sub_configs["vision_config"](
                hidden_size=128, intermediate_size=256, depth=0,
                num_heads=1, patch_size=16, spatial_merge_size=2,
                temporal_patch_size=2, in_channels=3,
            )'''

    if old in content:
        content = content.replace(old, new)
        with open(CONFIG_PATH, 'w') as f:
            f.write(content)
        print("[patch] config: vision_config default -> None (text-only safe)")
    else:
        print("[patch] config: Could not find vision_config default pattern")


if __name__ == "__main__":
    print("=" * 55)
    print("vLLM Patch: Qwen3.5 MoE text-only shim v3")
    print("=" * 55)
    patch_registry()
    patch_add_text_only_shim()
    patch_processing_info()
    patch_config_vision_default()
    print("=" * 55)
    print("Patch complete")
