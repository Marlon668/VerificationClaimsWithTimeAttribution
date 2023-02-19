import de.unihd.dbs.heideltime.standalone.DocumentType;
import de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone;
import de.unihd.dbs.heideltime.standalone.OutputType;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class NormalisePublicationTimeClaim {

    /*
    normalise the publication time of the claims
    input is a string of publication times with where each entry is split by a tab ("\t") and contains first the id and
    second their unnormalised publication time
    output: timex normalisation files in xml format saved in the folder ProcessedDates
    @arg 1 path to file containing unnormalised times
    @arg 2 path to folder ProcessedDates where the timex normalisation of the publication dates are saved to
     */
    public static void main(String[] args) throws Exception {

        List<String>processedDates = new ArrayList<>();
        int processed = 0;
        File file = new File(args[0]);
        Files.createDirectories(Paths.get(args[1]));
        Scanner sc = new Scanner(file);
        HeidelTimeStandalone heideltime = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NARRATIVES, OutputType.TIMEML, "config.props");
        while(sc.hasNextLine()) {
            String[] line = sc.nextLine().split("\t");
            String process = heideltime.process(line[1]);
            FileWriter myWriter = new FileWriter(args[1]+"/"+line[0]+".xml");
            myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') +1)+1));
            myWriter.close();
            processed++;
            System.out.println("Processed : " + processed);
        }
    }
}