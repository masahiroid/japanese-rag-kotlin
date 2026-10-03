plugins {
    id("com.android.application")
}

android {
    namespace = "jp.masahirocom.ragkit.sample"
    compileSdk = 36
    defaultConfig {
        applicationId = "jp.masahirocom.ragkit.sample"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += "json" }
}

dependencies {
    implementation(project(":litert"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}
