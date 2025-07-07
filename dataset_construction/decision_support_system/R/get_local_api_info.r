# Install if needed
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

package_name <- "RMCDA"  # Replace with the package you want to analyze

library(jsonlite)
library(package_name, character.only = TRUE)

clean_doc_readable <- function(text) {
  # Replace lines that look like "_L_e_t_t_e_r_s" with "Letters"
  # The pattern is underscore + letter + underscore + letter ... etc.

  # Function to fix one line
  fix_line <- function(line) {
    # If line contains underscores and letters spaced, remove underscores and spaces between letters
    # Pattern: underscore + letter, repeated, possibly separated by spaces

    # We'll replace patterns like "_P_l_o_t" or "_D_e_s_c_r_i_p_t_i_o_n"
    # by extracting only the letters, skipping underscores and spaces between letters

    if (grepl("_[A-Za-z](_[A-Za-z])*", line)) {
      # Remove underscores and spaces between letters for these words

      # split line into tokens by spaces
      tokens <- strsplit(line, " ")[[1]]

      # For each token, if it contains underscores, remove underscores and join letters
      tokens_fixed <- sapply(tokens, function(token) {
        if (grepl("_", token)) {
          # remove underscores and join letters
          letters_only <- gsub("_", "", token)
          return(letters_only)
        } else {
          return(token)
        }
      }, USE.NAMES = FALSE)

      return(paste(tokens_fixed, collapse = " "))
    } else {
      return(line)
    }
  }

  # Process each line
  lines <- unlist(strsplit(text, "\n"))
  lines_fixed <- sapply(lines, fix_line, USE.NAMES = FALSE)
  paste(lines_fixed, collapse = "\n")
}

extract_example <- function(topic, package) {
  tryCatch(
    {
      cat("Extracting example for:", topic, "\n")
      # Get help file reference
      hfile <- utils::help(topic, package <- package)

      if (length(hfile) == 0) {
        warning("No help file found")
        return(NULL)
      }

      rd_or_path <- utils:::.getHelpFile(hfile[[1]])

      if (is.character(rd_or_path) && length(rd_or_path) == 1 && file.exists(rd_or_path)) {
        # rd_or_path is a file path, parse it
        rd <- tools::parse_Rd(rd_or_path)
      } else if (is.list(rd_or_path)) {
        # rd_or_path is already parsed Rd content
        rd <- rd_or_path
      } else {
        stop("Help file format not recognized")
      }
      
      # Extract the examples section
      examples <- unlist(
        lapply(rd, function(tag) {
          if (attr(tag, "Rd_tag") == "\\examples") {
            return(paste(sapply(tag, function(x) {
              if (is.character(x)) {
                return(x)
              } else if (is.list(x)) {
                return(paste(unlist(x), collapse = ""))
              } else {
                return("")
              }
            }), collapse = ""))
          } else {
            return(NULL)
          }
        })
      )

      # Clean leading/trailing whitespace
      examples <- trimws(examples)

      if (length(examples) == 0 || examples == "") {
        message("No examples found.")
        return(NULL)
      }

      return(examples)

    },
    error = function(e) {
      message("Error extracting example for: ", topic, " - ", e$message)
      return(NULL)
    }
  )
}

extract_doc <- function(topic, package) {
  tryCatch(
    {
      cat("Extracting documentation for:", topic, "\n")
      # Get help file reference
      hfile <- utils::help(topic, package <- package)

      if (length(hfile) == 0) {
        warning("No help file found")
        return(NULL)
      }

      rd_or_path <- utils:::.getHelpFile(hfile[[1]])

      if (is.character(rd_or_path) && length(rd_or_path) == 1 && file.exists(rd_or_path)) {
        # rd_or_path is a file path, parse it
        rd <- tools::parse_Rd(rd_or_path)
      } else if (is.list(rd_or_path)) {
        # rd_or_path is already parsed Rd content
        rd <- rd_or_path
      } else {
        stop("Help file format not recognized")
      }

      doc_lines <- capture.output(tools::Rd2txt(rd))
      doc_string <- paste(doc_lines, collapse = "\n")

      # Remove all backspace characters (\b)
      cleaned_doc <- clean_doc_readable(gsub("\b", "", doc_string, fixed = TRUE))

      cleaned_doc
    },
    error = function(e) {
      message("Error extracting documentation for: ", topic, " - ", e$message)
      return(NULL)
    }
  )
}

extract_source_code <- function(topic, package) {
  tryCatch(
    {
      cat("Extracting source code for:", topic, "\n")
      # Get the function from the package namespace
      f <- get(topic, envir = asNamespace(package))

      if (!is.function(f)) {
        warning("Not a function: ", topic)
        return(NULL)
      }

      # Get the source code
      src <- paste(deparse(f), collapse = "\n")
      # Remove leading and trailing whitespace
      src <- trimws(src)


      # Remove any backspace characters (\b)
      src <- gsub("\b", "", src, fixed = TRUE)

      src
    },
    error = function(e) {
      message("Error extracting source code for: ", topic, " - ", e$message)
      return(NULL)
    }
  )
}

# List all exported functions
exports <- getNamespaceExports(package_name)

# Main list
api_list <- list()

for (fname in exports) {
  f <- tryCatch(get(fname, envir = asNamespace(package_name)), error = function(e) NULL)
  if (!is.function(f)) next

  # Extract parameter names
  formals_list <- tryCatch(as.list(formals(f)), error = function(e) NULL)
  param_list <- if (!is.null(formals_list)) {
    sapply(names(formals_list), function(arg) {
      val <- formals_list[[arg]]

      # Protect against missing values or expressions
      if (missing(val) || is.symbol(val)) {
        return(arg)
      } else {
        val_str <- tryCatch(paste(deparse(val), collapse = ""), error = function(e) "")
        return(paste0(arg, " = ", val_str))
      }
    }, USE.NAMES = FALSE)
  } else {
    list()
  }


  # Extract documentation
  doc <- extract_doc(fname, package_name)

  # Extract example
  example <- extract_example(fname, package_name)

  # Extract source code
  src <- extract_source_code(fname, package_name)

  signature <- paste0("(", paste(param_list, collapse = ", "), ")")

  # Build final structure
  api_list[[fname]] <- list(
    package = package_name,
    name = fname,
    signature = signature,
    docstring = doc,
    # source code is name <- src
    source_code = paste0(fname, " <- ", src),
    example_code = example
  )
}

# Write to JSON
write_json(api_list, 
           path = paste0(package_name, "_raw_api.json"),
           pretty = TRUE,
           auto_unbox = TRUE)
cat("✅ Exported API specs and docs to: ", paste0(package_name, "_raw_api.json"), "\n")
