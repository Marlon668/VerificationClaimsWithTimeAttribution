import de.unihd.dbs.heideltime.standalone.DocumentType;
import de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone;
import de.unihd.dbs.heideltime.standalone.OutputType;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

public class normalisePublicationTimeSnippets {

    /*
    normalise the publication time of the snippets
    input is path of folder containing files of publication times with where each entry is split by a tab ("\t") and contains first the id and
    second their unnormalised publication time
    output: timex normalisation files in xml format saved in the folder processedSnippets
    @arg 1 path to file containing unnormalised publication times of snippets
    @arg 2 path to folder processedSnippets where the timex normalisation of the publication dates are saved to
     */
    public static void main(String[] args) throws Exception {

        List<String>processedDates = new ArrayList<>();
        int processed = 0;
        final File folder= new File(args[0]);
        Files.createDirectories(Paths.get(args[1]));
        HashMap<String,List<String>> directories = listDirectoriesForFolder(folder);
        for(String directory:directories.keySet()){
            for(String snippet: directories.get(directory)){
                File file = new File(args[0] + "/"+directory+"/"+snippet);
                Scanner sc = new Scanner(file);
                HeidelTimeStandalone heideltime = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NARRATIVES, OutputType.TIMEML, "config.props");
                while(sc.hasNextLine()) {
                    String line = sc.nextLine();
                    String process = heideltime.process(line);
                    File dict = new File(args[1] + "/" +directory);
                    dict.mkdir();
                    FileWriter myWriter = new FileWriter(args[1] + "/" +directory+"/"+snippet+".xml");
                    myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') +1)+1));
                    myWriter.close();
                    processed++;
                    System.out.println("Processed : " + processed);
                }
            }

        }
    }

    public static HashMap<String,List<String>> listDirectoriesForFolder(final File folder) {
        HashMap<String,List<String>> directories = new HashMap<>();
        for (final File fileEntry : folder.listFiles()) {
            directories.put(fileEntry.getName(),listFilesForDirectory(fileEntry));
        }
        return directories;
    }

    public static List<String> listFilesForDirectory(final File folder) {
        List<String> files = new ArrayList<>();
        for (final File fileEntry : folder.listFiles()) {
            files.add(fileEntry.getName());
        }
        return files;
    }
}