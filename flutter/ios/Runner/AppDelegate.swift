import Flutter
import UIKit

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)
        // Register native ML plugin for macro classifier + wear estimator
        if let registrar = self.registrar(forPlugin: "SlcMlPlugin") {
            SlcMlPlugin.register(with: registrar)
        }
        // Register tile detector + ARKit depth plugin
        if let registrar = self.registrar(forPlugin: "SlcDetectorPlugin") {
            SlcDetectorPlugin.register(with: registrar)
        }
        // Register MiDaS-small monocular depth (non-LiDAR fallback)
        if let registrar = self.registrar(forPlugin: "SlcMidasPlugin") {
            SlcMidasPlugin.register(with: registrar)
        }
        return super.application(application,
                                  didFinishLaunchingWithOptions: launchOptions)
    }
}
