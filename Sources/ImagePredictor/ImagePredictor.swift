// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Vision
import UIKit

/// A convenience class that makes image classification predictions.
///
/// The Image Predictor creates and reuses an instance of a Core ML image classifier inside a ``VNCoreMLRequest``.
/// Each time it makes a prediction, the class:
/// - Creates a `VNImageRequestHandler` with an image
/// - Starts an image classification request for that image
/// - Converts the prediction results in a completion handler
/// - Updates the delegate's `predictions` property
/// - Tag: ImagePredictor
public final class ImagePredictor {
    
    /// The function signature the caller must provide as a completion handler.
    public typealias ImagePredictionHandler = (_ predictions: [Prediction]?) -> Void
    
    /// A dictionary of prediction handler functions, each keyed by its Vision request.
    private var predictionHandlers = [VNRequest: ImagePredictionHandler]()
    
    /// Stores a classification name and confidence for an image classifier's prediction.
    /// - Tag: Prediction
    public struct Prediction {
        /// The name of the object or scene the image classifier recognizes in an image.
        let classification: String
        /// The image classifier's confidence as a float.
        let confidence: Float
    }
    
    public init() {
        
    }
        
    /// Generates an image classification prediction for a photo.
    /// - Parameter photo: An image, typically of an object or a scene.
    /// - Tag: makePredictions
    public func makePredictions(for photo: UIImage, completionHandler: @escaping ImagePredictionHandler) throws {
        let orientation = CGImagePropertyOrientation(photo.imageOrientation)
        
        guard let photoImage = photo.cgImage else {
            print("Photo doesn't have underlying CGImage.")
            return
        }
        
        guard let imageClassificationRequest = createImageClassificationRequest() else {
            return
        }
        
        predictionHandlers[imageClassificationRequest] = completionHandler
        
        let handler = VNImageRequestHandler(cgImage: photoImage, orientation: orientation)
        let requests: [VNRequest] = [imageClassificationRequest]
        
        // Start the image classification request.
        try handler.perform(requests)
    }

    /// Async generates an image classification prediction for a photo.
    /// - Parameter photo: An image, typically of an object or a scene.
    /// - Tag: asyncMakePredictions
    @available(iOS 13, *)
    public func makePredictions(for photo: UIImage) async throws -> [Prediction]? {
        try await withCheckedThrowingContinuation { continuation in
            do {
                try makePredictions(for: photo) { predictions in
                    continuation.resume(returning: predictions)
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    /// The completion handler method that Vision calls when it completes a request.
    /// - Parameters:
    ///   - request: A Vision request.
    ///   - error: An error if the request produced an error; otherwise `nil`.
    ///
    ///   The method checks for errors and validates the request's results.
    /// - Tag: visionRequestHandler
    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
        // Remove the caller's handler from the dictionary and keep a reference to it.
        guard let predictionHandler = predictionHandlers.removeValue(forKey: request) else {
            print("Every request must have a prediction handler.")
            return
        }
        
        // Start with a `nil` value in case there's a problem.
        var predictions: [Prediction]? = nil
        
        // Call the client's completion handler after the method returns.
        defer {
            // Send the predictions back to the client.
            predictionHandler(predictions)
        }
        
        // Check for an error first.
        if let error {
            print("Vision image classification error...\n\n\(error.localizedDescription)")
            return
        }
        
        // Check that the results aren't `nil`.
        if request.results == nil {
            print("Vision request had no results.")
            return
        }
        
        // Cast the request's results as an `VNClassificationObservation` array.
        guard let observations = request.results as? [VNClassificationObservation] else {
            // Image classifiers, like MobileNet, only produce classification observations.
            // However, other Core ML model types can produce other observations.
            // For example, a style transfer model produces `VNPixelBufferObservation` instances.
            print("VNRequest produced the wrong result type: \(type(of: request.results)).")
            return
        }
        
        // Create a prediction array from the observations.
        predictions = observations.map { observation in
            // Convert each observation into an `ImagePredictor.Prediction` instance.
            Prediction(classification: observation.identifier,
                       confidence: observation.confidence
            )
        }
    }
    
    /// Generates a new request instance that uses the Image Predictor's image classifier model.
    private func createImageClassificationRequest() -> VNImageBasedRequest? {
        // Create an image classification request with an image classifier model.
        
        /// A common image classifier instance that all Image Predictor instances use to generate predictions.
        ///
        /// Share one ``VNCoreMLModel`` instance --- for each Core ML model file --- across the app,
        /// since each can be expensive in time and resources.
        let imageClassifier: VNCoreMLModel? = {
            // Use a default model configuration.
            let defaultConfig = MLModelConfiguration()
            
            // Create an instance of the image classifier's wrapper class.
            guard let imageClassifier = try? MobileNet(configuration: defaultConfig) else {
                print("App failed to create an image classifier model instance.")
                return nil
            }
            
            // Create a Vision instance using the image classifier's model instance.
            guard let imageClassifierVisionModel = try? VNCoreMLModel(for: imageClassifier.model) else {
                print("App failed to create a `VNCoreMLModel` instance.")
                return nil
            }
            
            return imageClassifierVisionModel
        }()
        
        guard let imageClassifier else {
            return nil
        }
        
        let imageClassificationRequest = VNCoreMLRequest(model: imageClassifier, completionHandler: visionRequestHandler)
        imageClassificationRequest.imageCropAndScaleOption = .centerCrop
        return imageClassificationRequest
    }
}
