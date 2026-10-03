plugins {
    id("com.android.library")
}

android {
    namespace = "jp.masahirocom.ragkit.litert"
    compileSdk = 36
    defaultConfig { minSdk = 26 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    api(project(":core"))
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
}
