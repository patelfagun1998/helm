from typing import Optional

from helm.benchmark.scenarios.scenario import Instance
from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.adaptation.adapters.multimodal.multiple_choice_joint_multimodal_adapter import (
    MultipleChoiceJointMultimodalAdapter,
)


class MultipleChoiceJointMultimodalChainOfThoughtAdapter(MultipleChoiceJointMultimodalAdapter):
    """
    Each `Instance` in a multimodal `Scenario` looks like this:

        <multimodal_input> -> <reference1>
                              <reference2>
                              <reference3> [correct]
                              <reference4>

        <instance_chain_of_thought>

    We can define a label (e.g., letter) for each reference:

        <global_prefix>
        <instructions>
        <input_prefix>
        <multimodal_input>           # train
        <input_suffix>
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        <output_prefix>
        <chain_of_thought_prefix>
        <instance_chain_of_thought>
        <chain_of_thought_suffix>
        <output>
        <output_suffix>

        <input_prefix>
        <multimodal_input>           # test
        <input_suffix>
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        <output_prefix>
        <chain_of_thought_prefix>
        <instance_chain_of_thought>
        <chain_of_thought_suffix>
        <output>
        <output_suffix>
        <global_suffix>

    In general, each example is:

        <input_prefix><multimodal_input><input_suffix><reference_prefixes[index]><reference> \
        <output_prefix><chain_of_thought_prefix><chain_of_thought><chain_of_thought_suffix><output><output_suffix>
    """

    def construct_example_multimodal_prompt(
        self, instance: Instance, include_output: bool, reference_index: Optional[int]
    ) -> MultimediaObject:
        """
        Returns a single example of the multimodal prompt with chain of thought support.
        `include_output` controls whether the gold output is included.
        """
        # Input
        assert instance.input.multimedia_content is not None
        result: MultimediaObject = instance.input.multimedia_content.add_textual_prefix(self.adapter_spec.input_prefix)
        result = result.add_textual_suffix(self.adapter_spec.input_suffix)

        # Include the references
        delimiter: str = ", "
        no_correct_references: str = "n/a"
        output: str = no_correct_references
        for reference_index, reference in enumerate(instance.references):
            prefix = self.get_reference_prefix(self.adapter_spec.reference_prefix, reference_index)

            if reference.output.multimedia_content is not None:
                reference_output_content: MultimediaObject = reference.output.multimedia_content
                reference_output_content = reference_output_content.add_textual_prefix(prefix)
                reference_output_content = reference_output_content.add_textual_suffix(
                    self.adapter_spec.reference_suffix
                )
                result = result.combine(reference_output_content)
            else:
                result = result.add_textual_suffix(prefix + reference.output.text + self.adapter_spec.reference_suffix)

            if reference.is_correct:
                if output == no_correct_references:
                    output = self.get_reference_prefix("A", reference_index)
                elif self.adapter_spec.multi_label:
                    output += delimiter
                    output += self.get_reference_prefix("A", reference_index)

        if include_output:
            # Add chain of thought if available
            chain_of_thought = instance.extra_data.get("chain_of_thought", "") if instance.extra_data else ""
            
            # Create the chain of thought block
            chain_of_thought_block = ""
            if chain_of_thought:
                chain_of_thought_block = (
                    self.adapter_spec.chain_of_thought_prefix + chain_of_thought + self.adapter_spec.chain_of_thought_suffix
                )
            
            # Combine chain of thought with the output
            output_text = chain_of_thought_block + output
            output_content: MultimediaObject = MultimediaObject([MediaObject(text=output_text, content_type="text/plain")])
            output_content = output_content.add_textual_prefix(self.adapter_spec.output_prefix)
            output_content = output_content.add_textual_suffix(self.adapter_spec.output_suffix)
            result = result.combine(output_content)
        else:
            result = result.add_textual_suffix(self.adapter_spec.output_prefix.rstrip())

        return result 