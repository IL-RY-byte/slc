package com.slc.scanner

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {

    private var mlPlugin:       SlcMlPlugin?       = null
    private var detectorPlugin: SlcDetectorPlugin? = null
    private var midasPlugin:    SlcMidasPlugin?    = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        mlPlugin       = SlcMlPlugin(applicationContext).also       { it.register(flutterEngine) }
        detectorPlugin = SlcDetectorPlugin(applicationContext).also { it.register(flutterEngine) }
        midasPlugin    = SlcMidasPlugin(applicationContext).also    { it.register(flutterEngine) }
    }

    override fun onDestroy() {
        mlPlugin?.dispose()
        detectorPlugin?.dispose()
        midasPlugin?.dispose()
        super.onDestroy()
    }
}
