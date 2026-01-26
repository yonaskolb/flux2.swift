import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2KleinPipelineTests: XCTestCase {
  private var repoRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  func testTinyKleinPipelineRuns() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("klein_expected.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let expectedPromptEmbeds = try inputsReader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try inputsReader.tensor(named: "text_ids").asType(.int32)
    let expectedLatentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)
    let steps = Int(try inputsReader.tensor(named: "num_inference_steps").asType(.int32).asArray(Int32.self).first ?? 0)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )
    let output = try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      height: height,
      width: width,
      numInferenceSteps: steps,
      latents: latents
    )

    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))
    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.promptEmbeds, expectedPromptEmbeds, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(output.textIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
    XCTAssertEqual(output.latentIds.asType(.int32).asArray(Int32.self), expectedLatentIds.asArray(Int32.self))

    XCTAssertEqual(output.decoded.ndim, 4)
    let batch = output.decoded.dim(0)
    XCTAssertEqual(output.decoded.dim(1), 3)
    XCTAssertEqual(output.decoded.dim(2), height)
    XCTAssertEqual(output.decoded.dim(3), width)

    XCTAssertEqual(output.promptEmbeds.dim(0), batch)
    XCTAssertEqual(output.textIds.dim(0), batch)
    XCTAssertEqual(output.latentIds.dim(0), batch)
    XCTAssertEqual(output.packedLatents.dim(0), batch)
    XCTAssertNil(output.imageLatents)
    XCTAssertNil(output.imageLatentIds)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })
  }

  func testTinyKleinPipelineCFG() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline_cfg")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_cfg_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("klein_cfg_expected.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let negativeInputIds = try inputsReader.tensor(named: "negative_input_ids").asType(.int32)
    let negativeAttentionMask = try inputsReader.tensor(named: "negative_attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let expectedPromptEmbeds = try inputsReader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try inputsReader.tensor(named: "text_ids").asType(.int32)
    let expectedLatentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)
    let steps = Int(try inputsReader.tensor(named: "num_inference_steps").asType(.int32).asArray(Int32.self).first ?? 0)
    let guidanceScale = Float(try inputsReader.tensor(named: "guidance_scale").asType(.float32).asArray(Float32.self).first ?? 1)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )
    let output = try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      negativeInputIds: negativeInputIds,
      negativeAttentionMask: negativeAttentionMask,
      height: height,
      width: width,
      numInferenceSteps: steps,
      latents: latents,
      guidanceScale: guidanceScale
    )

    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))
    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.promptEmbeds, expectedPromptEmbeds, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(output.textIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
    XCTAssertEqual(output.latentIds.asType(.int32).asArray(Int32.self), expectedLatentIds.asArray(Int32.self))

    XCTAssertEqual(output.decoded.ndim, 4)
    let batch = output.decoded.dim(0)
    XCTAssertEqual(output.decoded.dim(1), 3)
    XCTAssertEqual(output.decoded.dim(2), height)
    XCTAssertEqual(output.decoded.dim(3), width)

    XCTAssertEqual(output.promptEmbeds.dim(0), batch)
    XCTAssertEqual(output.textIds.dim(0), batch)
    XCTAssertEqual(output.latentIds.dim(0), batch)
    XCTAssertEqual(output.packedLatents.dim(0), batch)
    XCTAssertNil(output.imageLatents)
    XCTAssertNil(output.imageLatentIds)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })
  }

  func testGenerateTokensThrowsOnNegativeInputShapeMismatch() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline_cfg")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_cfg_inputs.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let negativeInputIds = try inputsReader.tensor(named: "negative_input_ids").asType(.int32)
    let negativeAttentionMask = try inputsReader.tensor(named: "negative_attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )

    let truncatedLen = max(0, negativeInputIds.dim(1) - 1)
    let badNegativeInputIds = negativeInputIds[.ellipsis, ..<truncatedLen]
    XCTAssertThrowsError(try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      negativeInputIds: badNegativeInputIds,
      negativeAttentionMask: negativeAttentionMask,
      height: height,
      width: width,
      numInferenceSteps: 1,
      latents: latents,
      guidanceScale: 2.0
    )) { error in
      guard case Flux2KleinPipelineError.negativeInputIdsShapeMismatch(expected: _, got: _) = error else {
        XCTFail("Unexpected error \(error)")
        return
      }
    }
  }

  func testGenerateTokensThrowsOnPartialNegativeTokens() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline_cfg")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_cfg_inputs.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let negativeInputIds = try inputsReader.tensor(named: "negative_input_ids").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )

    XCTAssertThrowsError(try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      negativeInputIds: negativeInputIds,
      negativeAttentionMask: nil,
      height: height,
      width: width,
      numInferenceSteps: 1,
      latents: latents,
      guidanceScale: 2.0
    )) { error in
      guard case Flux2KleinPipelineError.invalidNegativeTokens = error else {
        XCTFail("Unexpected error \(error)")
        return
      }
    }
  }

  func testTinyKleinImagePipelineRuns() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_image_pipeline")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_image_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("klein_image_expected.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let image = try inputsReader.tensor(named: "image")
    let expectedPromptEmbeds = try inputsReader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try inputsReader.tensor(named: "text_ids").asType(.int32)
    let expectedLatentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)
    let steps = Int(try inputsReader.tensor(named: "num_inference_steps").asType(.int32).asArray(Int32.self).first ?? 0)
    let imageIdScale = Int((try? inputsReader.tensor(named: "image_id_scale").asType(.int32).asArray(Int32.self).first) ?? 10)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )
    let output = try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      height: height,
      width: width,
      numInferenceSteps: steps,
      latents: latents,
      images: [image],
      imageIdScale: imageIdScale
    )
    guard let actualImageLatents = output.imageLatents,
          let actualImageLatentIds = output.imageLatentIds else {
      XCTFail("Expected image-conditioned pipeline output to include image latents.")
	      return
	    }

    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))
    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    let expectedImageLatents = try expectedReader.tensor(named: "image_latents").asType(.float32)
    let expectedImageLatentIds = try expectedReader.tensor(named: "image_latent_ids").asType(.int32)
    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.promptEmbeds, expectedPromptEmbeds, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(output.textIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
    XCTAssertEqual(output.latentIds.asType(.int32).asArray(Int32.self), expectedLatentIds.asArray(Int32.self))
    TestHelpers.assertAllClose(actualImageLatents, expectedImageLatents, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(actualImageLatentIds.asType(.int32).asArray(Int32.self), expectedImageLatentIds.asArray(Int32.self))

    XCTAssertEqual(output.decoded.ndim, 4)
    let batch = output.decoded.dim(0)
    XCTAssertEqual(output.decoded.dim(1), 3)
    XCTAssertEqual(output.decoded.dim(2), height)
    XCTAssertEqual(output.decoded.dim(3), width)

    XCTAssertEqual(actualImageLatents.ndim, 3)
    XCTAssertEqual(actualImageLatentIds.ndim, 3)
    XCTAssertEqual(actualImageLatents.dim(0), batch)
    XCTAssertEqual(actualImageLatentIds.dim(0), batch)
    XCTAssertEqual(actualImageLatents.dim(1), actualImageLatentIds.dim(1))
    XCTAssertEqual(actualImageLatentIds.dim(2), 4)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })
  }

  func testTinyKleinImagePipelineCFGRuns() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_image_pipeline_cfg")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_image_cfg_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("klein_image_cfg_expected.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let negativeInputIds = try inputsReader.tensor(named: "negative_input_ids").asType(.int32)
    let negativeAttentionMask = try inputsReader.tensor(named: "negative_attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let image = try inputsReader.tensor(named: "image")
    let expectedPromptEmbeds = try inputsReader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try inputsReader.tensor(named: "text_ids").asType(.int32)
    let expectedLatentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)
    let steps = Int(try inputsReader.tensor(named: "num_inference_steps").asType(.int32).asArray(Int32.self).first ?? 0)
    let imageIdScale = Int((try? inputsReader.tensor(named: "image_id_scale").asType(.int32).asArray(Int32.self).first) ?? 10)
    let guidanceScale = Float(try inputsReader.tensor(named: "guidance_scale").asType(.float32).asArray(Float32.self).first ?? 1)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )
    let output = try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      negativeInputIds: negativeInputIds,
      negativeAttentionMask: negativeAttentionMask,
      height: height,
      width: width,
      numInferenceSteps: steps,
      latents: latents,
      guidanceScale: guidanceScale,
      images: [image],
      imageIdScale: imageIdScale
    )
    guard let actualImageLatents = output.imageLatents,
          let actualImageLatentIds = output.imageLatentIds else {
      XCTFail("Expected image-conditioned CFG output to include image latents.")
	      return
	    }

    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))
    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    let expectedImageLatents = try expectedReader.tensor(named: "image_latents").asType(.float32)
    let expectedImageLatentIds = try expectedReader.tensor(named: "image_latent_ids").asType(.int32)
    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.promptEmbeds, expectedPromptEmbeds, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(output.textIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
    XCTAssertEqual(output.latentIds.asType(.int32).asArray(Int32.self), expectedLatentIds.asArray(Int32.self))
    TestHelpers.assertAllClose(actualImageLatents, expectedImageLatents, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(actualImageLatentIds.asType(.int32).asArray(Int32.self), expectedImageLatentIds.asArray(Int32.self))

    XCTAssertEqual(output.decoded.ndim, 4)
    let batch = output.decoded.dim(0)
    XCTAssertEqual(output.decoded.dim(1), 3)
    XCTAssertEqual(output.decoded.dim(2), height)
    XCTAssertEqual(output.decoded.dim(3), width)

    XCTAssertEqual(actualImageLatents.ndim, 3)
    XCTAssertEqual(actualImageLatentIds.ndim, 3)
    XCTAssertEqual(actualImageLatents.dim(0), batch)
    XCTAssertEqual(actualImageLatentIds.dim(0), batch)
    XCTAssertEqual(actualImageLatents.dim(1), actualImageLatentIds.dim(1))
    XCTAssertEqual(actualImageLatentIds.dim(2), 4)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })
  }

  func testTokensGuidanceRequiresNegativeTokens() throws {
    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_inputs.safetensors")

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let height = Int(try inputsReader.tensor(named: "height").asType(.int32).asArray(Int32.self).first ?? 0)
    let width = Int(try inputsReader.tensor(named: "width").asType(.int32).asArray(Int32.self).first ?? 0)

    let pipeline = try Flux2KleinPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [0],
      loadTokenizer: false
    )

    XCTAssertThrowsError(
      try pipeline.generateTokens(
        inputIds: inputIds,
        attentionMask: attentionMask,
        height: height,
        width: width,
        numInferenceSteps: 1,
        latents: latents,
        guidanceScale: 2.0
      )
    )
  }
}
