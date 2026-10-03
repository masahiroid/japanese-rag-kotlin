rootProject.name = "japanese-rag-kotlin"
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
include(":core")
include(":litert")
